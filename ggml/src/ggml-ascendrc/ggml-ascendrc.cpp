#include "ggml-impl.h"
#include "ggml-ascendrc.h"
#include "ggml-backend-impl.h"
#include "ggml.h"

#include <future>
#include <vector>
#include <cstring>
#include <iostream>

#include <acl/acl.h>
#include <ascblas.h>

#ifdef TRACY_ENABLE
#include "tracy/Tracy.hpp"
#endif

#define GGML_CANN_MAX_STREAMS 8

// Macro function for unwinding acl errors.
#define ACL_CHECK(status)                                                                                              \
    do {                                                                                                               \
        aclError error = status;                                                                                       \
        if (error != ACL_ERROR_NONE) {                                                                                 \
            std::cerr << __FILE__ << ":" << __LINE__ << " aclError:" << error << std::endl;                            \
        }                                                                                                              \
    } while (0)

thread_local int g_current_ascendrc_device = -1;

void ggml_ascendrc_set_device(const int32_t device) {
    // int current_device = -1;
    // Note: In some CANN versions, if no device has been set yet,
    //       aclrtGetDevice(&current_device) may return 0 by default.
    // aclrtGetDevice(&current_device);

    // If the current device is already the target one, no need to switch.
    if (device == g_current_ascendrc_device) {
        return;
    }

    // Switch to the new device.
    ACL_CHECK(aclrtSetDevice(device));

    // Update the global device record.
    g_current_ascendrc_device = device;
}

struct ggml_backend_ascendrc_context {
    int32_t     device;               /**< Device ID. */
    std::string name;                 /**< Name of the device. */
    std::string description;          /**< Description of the device. */
    int n_threads = GGML_DEFAULT_N_THREADS;
    std::unique_ptr<char[]> work_data;
    size_t work_size = 0;
    aclrtStream streams[GGML_CANN_MAX_STREAMS] = { nullptr };
    aclrtStream stream(int stream) {
        if (streams[stream] == nullptr) {
            // If the device is not set here, destroying the stream later may cause a mismatch
            // between the thread contexts where the stream was created and destroyed.
            // However, I printed the device_id, thread_id, and stream, and they are all consistent.
            ACL_CHECK(aclrtSetDevice(device));
            ACL_CHECK(aclrtCreateStream(&streams[stream]));
        }
        return streams[stream];
    }

    aclrtStream stream() { return stream(0); }

#ifndef GGML_USE_OPENMP
    std::vector<std::future<void>> tasks;
#endif

    explicit ggml_backend_ascendrc_context(int device) : device(device), name("AscendRC" + std::to_string(device)) {
        ggml_ascendrc_set_device(device);
        description = aclrtGetSocName();
    }

    ~ggml_backend_ascendrc_context() {
        ggml_ascendrc_set_device(device);

        for (int i = 0; i < GGML_CANN_MAX_STREAMS; ++i) {
            if (streams[i] != nullptr) {
                ACL_CHECK(aclrtDestroyStream(streams[i]));
            }
        }
    }
};

static void ggml_backend_ascendrc_mul_mat(ggml_backend_ascendrc_context * ctx, struct ggml_tensor * dst) {
    ZoneScopedNC("ascend_mul_mat", tracy::Color::Purple);
    ZoneValue(dst->ne[0] * dst->ne[1]);

    ggml_ascendrc_set_device(ctx->device);

    const struct ggml_tensor * src0 = dst->src[0];
    const struct ggml_tensor * src1 = dst->src[1];

    GGML_TENSOR_BINARY_OP_LOCALS

    const enum ggml_type type = src0->type;

    GGML_ASSERT(ne0 == ne01);
    GGML_ASSERT(ne1 == ne11);
    GGML_ASSERT(ne2 == ne12);
    GGML_ASSERT(ne3 == ne13);

    // we don't support permuted src0 or src1
    GGML_ASSERT(nb00 == ggml_type_size(type));
    GGML_ASSERT(nb10 == ggml_type_size(src1->type));

    // dst cannot be transposed or permuted
    GGML_ASSERT(nb0 == sizeof(float));
    GGML_ASSERT(nb0 <= nb1);
    GGML_ASSERT(nb1 <= nb2);
    GGML_ASSERT(nb2 <= nb3);

    // broadcast factors
    const int64_t r2 = ne12/ne02;
    const int64_t r3 = ne13/ne03;

    // Map ggml type to aclDataType
    aclDataType acl_dtype;
    switch (type) {
        case GGML_TYPE_F32:
            acl_dtype = ACL_FLOAT;
            break;
        case GGML_TYPE_F16:
            acl_dtype = ACL_FLOAT16;
            break;
        default:
            GGML_ABORT("Unsupported type for AscendBLAS");
    }

    const float alpha = 1.0f;
    const float beta = 0.0f;

    for (int64_t i13 = 0; i13 < ne13; i13++) {
        for (int64_t i12 = 0; i12 < ne12; i12++) {
            const int64_t i03 = i13/r3;
            const int64_t i02 = i12/r2;

            const void * x = (char *) src0->data + i02*nb02 + i03*nb03;
            const float * y_f32 = (float *) ((char *) src1->data + i12*nb12 + i13*nb13);
            float * d_f32 = (float *) ((char *) dst->data + i12*nb2 + i13*nb3);

            if (type == GGML_TYPE_F16) {
                // For FP16: convert src1 (FP32) to FP16, compute, then convert dst back to FP32
                // Allocate temporary buffers on host
                size_t ne_src1 = ne1 * ne10;
                size_t ne_dst = ne1 * ne01;

                // Use work_data buffer for temporary FP16 storage
                size_t needed_size = (ne_src1 + ne_dst) * sizeof(uint16_t);
                if (ctx->work_size < needed_size) {
                    ctx->work_data.reset(new char[needed_size]);
                    ctx->work_size = needed_size;
                }

                uint16_t * src1_fp16 = (uint16_t *)ctx->work_data.get();

                {
                    ZoneScopedN("cpu_convert_f32_to_f16");
                    for (size_t i=0;i<ne_src1;i++) {
                        src1_fp16[i] = GGML_FP32_TO_FP16(y_f32[i]);
                    }
                }

                {
                    // Call AscendBLAS Gemm with FP16
                    ZoneScopedN("ascendblas_gemm_fp16");
                    ascblasGemmEx(
                        ctx->stream(),
                        ASCBLAS_OP_N,
                        ASCBLAS_OP_T,
                        (int)ne1, (int)ne01, (int)ne10,
                        &alpha,
                        src1_fp16, ACL_FLOAT16, (int)ne10,
                        x, ACL_FLOAT16, (int)ne00,
                        &beta,
                        d_f32, ACL_FLOAT, (int)ne01);
                }
            } else {
                ZoneScopedN("ascendblas_gemm_fp32");
                // FP32 path - direct call
                ascblasGemmEx(
                    ctx->stream(),
                    ASCBLAS_OP_N,
                    ASCBLAS_OP_T,
                    (int)ne1, (int)ne01, (int)ne10,
                    &alpha,
                    y_f32, ACL_FLOAT, (int)ne10,
                    x, ACL_FLOAT, (int)ne00,
                    &beta,
                    d_f32, ACL_FLOAT, (int)ne01);
            }
        }
    }
}

// backend interface

static const char * ggml_backend_ascendrc_get_name(ggml_backend_t backend) {
    ggml_backend_ascendrc_context * ascendrc_ctx = (ggml_backend_ascendrc_context *) backend->context;

    return ascendrc_ctx->name.c_str();
}

static void ggml_backend_ascendrc_free(ggml_backend_t backend) {
    ggml_backend_ascendrc_context * ctx = (ggml_backend_ascendrc_context *)backend->context;
    delete ctx;
    delete backend;
}

static enum ggml_status ggml_backend_ascendrc_graph_compute(ggml_backend_t backend, struct ggml_cgraph * cgraph) {
    ggml_backend_ascendrc_context * ctx = (ggml_backend_ascendrc_context *)backend->context;

    for (int i = 0; i < cgraph->n_nodes; i++) {
        struct ggml_tensor * node = cgraph->nodes[i];

        if ((node->flags & GGML_TENSOR_FLAG_COMPUTE) == 0) {
            continue;
        }

        switch (node->op) {
            case GGML_OP_MUL_MAT:
                ggml_backend_ascendrc_mul_mat(ctx, node);
                break;

            case GGML_OP_NONE:
            case GGML_OP_RESHAPE:
            case GGML_OP_VIEW:
            case GGML_OP_PERMUTE:
            case GGML_OP_TRANSPOSE:
                break;

            default:
                GGML_ABORT("%s: unsupported op %s\n", __func__, ggml_op_desc(node));
        }
    }

    return GGML_STATUS_SUCCESS;

    GGML_UNUSED(backend);
}

static struct ggml_backend_i ascendrc_backend_i = {
    /* .get_name                = */ ggml_backend_ascendrc_get_name,
    /* .free                    = */ ggml_backend_ascendrc_free,
    /* .set_tensor_async        = */ NULL,
    /* .get_tensor_async        = */ NULL,
    /* .cpy_tensor_async        = */ NULL,
    /* .synchronize             = */ NULL,
    /* .graph_plan_create       = */ NULL,
    /* .graph_plan_free         = */ NULL,
    /* .graph_plan_update       = */ NULL,
    /* .graph_plan_compute      = */ NULL,
    /* .graph_compute           = */ ggml_backend_ascendrc_graph_compute,
    /* .event_record            = */ NULL,
    /* .event_wait              = */ NULL,
    /* .graph_optimize          = */ NULL,
};

static ggml_guid_t ggml_backend_ascendrc_guid(void) {
    static ggml_guid guid = { 0x50, 0xb9, 0x22, 0xe1, 0x40, 0xbf, 0x40, 0x0c, 0x86, 0xaf, 0x61, 0x89, 0x1b, 0xca, 0xa5, 0xb7 };

    return &guid;
}

ggml_backend_t ggml_backend_ascendrc_init(void) {
    aclInit(nullptr);

    ggml_backend_ascendrc_context * ctx = new ggml_backend_ascendrc_context(0);

    ggml_backend_t backend = new ggml_backend {
        /* .guid    = */ ggml_backend_ascendrc_guid(),
        /* .iface   = */ ascendrc_backend_i,
        /* .device  = */ ggml_backend_reg_dev_get(ggml_backend_ascendrc_reg(), 0),
        /* .context = */ ctx,
    };

    return backend;
}

bool ggml_backend_is_ascendrc(ggml_backend_t backend) {
    return backend != NULL && ggml_guid_matches(backend->guid, ggml_backend_ascendrc_guid());
}

void ggml_backend_ascendrc_set_n_threads(ggml_backend_t backend_ascendrc, int n_threads) {
    GGML_ASSERT(ggml_backend_is_ascendrc(backend_ascendrc));

    ggml_backend_ascendrc_context * ctx = (ggml_backend_ascendrc_context *)backend_ascendrc->context;
    ctx->n_threads = n_threads;
}

// device interface

static const char * ggml_backend_ascendrc_device_get_name(ggml_backend_dev_t dev) {
    return "AscendRC";

    GGML_UNUSED(dev);
}

static const char * ggml_backend_ascendrc_device_get_description(ggml_backend_dev_t dev) {
    return "fallback for chips without full CANN ops";

    GGML_UNUSED(dev);
}

static void ggml_backend_ascendrc_device_get_memory(ggml_backend_dev_t dev, size_t * free, size_t * total) {
    // no memory to report
    *free  = 0;
    *total = 0;

    GGML_UNUSED(dev);
}

static enum ggml_backend_dev_type ggml_backend_ascendrc_device_get_type(ggml_backend_dev_t dev) {
    return GGML_BACKEND_DEVICE_TYPE_ACCEL;

    GGML_UNUSED(dev);
}

static void ggml_backend_ascendrc_device_get_props(ggml_backend_dev_t dev, struct ggml_backend_dev_props * props) {
    props->name        = ggml_backend_ascendrc_device_get_name(dev);
    props->description = ggml_backend_ascendrc_device_get_description(dev);
    props->type        = ggml_backend_ascendrc_device_get_type(dev);
    ggml_backend_ascendrc_device_get_memory(dev, &props->memory_free, &props->memory_total);
    props->caps = {
        /* .async                 = */ false,
        /* .host_buffer           = */ false,
        /* .buffer_from_host_ptr  = */ true,
        /* .events                = */ false,
    };
}

static ggml_backend_t ggml_backend_ascendrc_device_init_backend(ggml_backend_dev_t dev, const char * params) {
    return ggml_backend_ascendrc_init();

    GGML_UNUSED(dev);
    GGML_UNUSED(params);
}

static ggml_backend_buffer_type_t ggml_backend_ascendrc_device_get_buffer_type(ggml_backend_dev_t dev) {
    return ggml_backend_cpu_buffer_type();

    GGML_UNUSED(dev);
}

static ggml_backend_buffer_t ggml_backend_ascendrc_device_buffer_from_host_ptr(ggml_backend_dev_t dev, void * ptr, size_t size, size_t max_tensor_size) {
    return ggml_backend_cpu_buffer_from_ptr(ptr, size);

    GGML_UNUSED(dev);
    GGML_UNUSED(max_tensor_size);
}

static bool ggml_backend_ascendrc_device_supports_op(ggml_backend_dev_t dev, const struct ggml_tensor * op) {
    const struct ggml_tensor * src0 = op->src[0];
    const struct ggml_tensor * src1 = op->src[1];

    switch (op->op) {
        case GGML_OP_NONE:
        case GGML_OP_RESHAPE:
        case GGML_OP_VIEW:
        case GGML_OP_PERMUTE:
        case GGML_OP_TRANSPOSE:
            return true;

        case GGML_OP_MUL_MAT:
        {
            const struct ggml_tensor * src0 = op->src[0];
            const struct ggml_tensor * src1 = op->src[1];

            const int64_t ne10 = src1->ne[0];

            const int64_t ne0 = op->ne[0];
            const int64_t ne1 = op->ne[1];

            // TODO: find the optimal value
            const int64_t min_batch = 1;

            return ggml_is_contiguous(src0) &&
                   ggml_is_contiguous(src1) &&
                   src1->type == GGML_TYPE_F32 &&
                   (ne0 >= min_batch && ne1 >= min_batch && ne10 >= min_batch) &&
                   (src0->type == GGML_TYPE_F32 || src0->type == GGML_TYPE_F16);
        }

        default:
            return false;

    }

    GGML_UNUSED(dev);
}

static bool ggml_backend_ascendrc_device_supports_buft(ggml_backend_dev_t dev, ggml_backend_buffer_type_t buft) {
    return ggml_backend_buft_is_host(buft);

    GGML_UNUSED(dev);
}

static const struct ggml_backend_device_i ggml_backend_ascendrc_device_i = {
    /* .get_name             = */ ggml_backend_ascendrc_device_get_name,
    /* .get_description      = */ ggml_backend_ascendrc_device_get_description,
    /* .get_memory           = */ ggml_backend_ascendrc_device_get_memory,
    /* .get_type             = */ ggml_backend_ascendrc_device_get_type,
    /* .get_props            = */ ggml_backend_ascendrc_device_get_props,
    /* .init_backend         = */ ggml_backend_ascendrc_device_init_backend,
    /* .get_buffer_type      = */ ggml_backend_ascendrc_device_get_buffer_type,
    /* .get_host_buffer_type = */ NULL,
    /* .buffer_from_host_ptr = */ ggml_backend_ascendrc_device_buffer_from_host_ptr,
    /* .supports_op          = */ ggml_backend_ascendrc_device_supports_op,
    /* .supports_buft        = */ ggml_backend_ascendrc_device_supports_buft,
    /* .offload_op           = */ NULL,
    /* .event_new            = */ NULL,
    /* .event_free           = */ NULL,
    /* .event_synchronize    = */ NULL,
};

// backend reg interface

static const char * ggml_backend_ascendrc_reg_get_name(ggml_backend_reg_t reg) {
    return "AscendRC";

    GGML_UNUSED(reg);
}

static size_t ggml_backend_ascendrc_reg_get_device_count(ggml_backend_reg_t reg) {
    return 1;

    GGML_UNUSED(reg);
}

static ggml_backend_dev_t ggml_backend_ascendrc_reg_get_device(ggml_backend_reg_t reg, size_t index) {
    GGML_ASSERT(index == 0);

    static ggml_backend_device ggml_backend_ascendrc_device = {
        /* .iface   = */ ggml_backend_ascendrc_device_i,
        /* .reg     = */ reg,
        /* .context = */ nullptr,
    };

    return &ggml_backend_ascendrc_device;

    GGML_UNUSED(reg);
    GGML_UNUSED(index);
}

static void * ggml_backend_ascendrc_get_proc_address(ggml_backend_reg_t reg, const char * name) {
    if (std::strcmp(name, "ggml_backend_set_n_threads") == 0) {
        return (void *)ggml_backend_ascendrc_set_n_threads;
    }
    return NULL;

    GGML_UNUSED(reg);
    GGML_UNUSED(name);
}

static const struct ggml_backend_reg_i ggml_backend_ascendrc_reg_i = {
    /* .get_name         = */ ggml_backend_ascendrc_reg_get_name,
    /* .get_device_count = */ ggml_backend_ascendrc_reg_get_device_count,
    /* .get_device       = */ ggml_backend_ascendrc_reg_get_device,
    /* .get_proc_address = */ ggml_backend_ascendrc_get_proc_address,
};

ggml_backend_reg_t ggml_backend_ascendrc_reg(void) {
    static struct ggml_backend_reg ggml_backend_ascendrc_reg = {
        /* .api_version = */ GGML_BACKEND_API_VERSION,
        /* .iface       = */ ggml_backend_ascendrc_reg_i,
        /* .context     = */ NULL,
    };

    return &ggml_backend_ascendrc_reg;
}

GGML_BACKEND_DL_IMPL(ggml_backend_ascendrc_reg)
