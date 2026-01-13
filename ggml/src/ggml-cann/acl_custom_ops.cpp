#include <cinttypes>
#include <cstdio>

#include "acl/acl_op_compiler.h"

#include "acl_tensor.h"
#include "common.h"
#include "acl_custom_ops.h"

#include "aclnn_mm_custom.h"
#include "aclnn_batch_matmul_custom.h"
#include "aclnn_matmul_custom.h"
#include "aclnn_cast_custom.h"
#include "aclnn_ops.h"

static void aclnn_cast_custom(ggml_backend_cann_context & ctx,
                       aclTensor *                 acl_src,
                       aclTensor *                 acl_dst) {
    GGML_CANN_CALL_ACLNN_OP(ctx, CastCustom, acl_src, acl_dst);
}

void print_tensor_shape(const ggml_tensor * tensor) {
    printf("Tensor shape: (");
    for (int i = 0; i < 4; i++) {
        printf("%" PRId64, tensor->ne[i]);
        if (i < 4 - 1) {
            printf(", ");
        }
    }
    printf(")\n");

    printf("\tTotal elements: %zu\n", ggml_nelements(tensor));
    printf("\tElement type: %s\n", ggml_type_name(tensor->type));
}

static void ggml_cann_mat_mul_custom_fp(ggml_backend_cann_context & ctx, ggml_tensor * dst) {
    ggml_tensor * weight = dst->src[0];  // weight
    ggml_tensor * input  = dst->src[1];  // input

    // when weight ne2 or ne3 is 1, aclnnMatmulGetWorkspaceSize will auto
    // broadcast, when weight ne2 or ne3 is not 1, weight need repeat.
    BCAST_MUL_MAT_SHAPE(input, weight, dst);

    int64_t n_dims = bcast_dims;
    if (bcast_input_ne[3] == bcast_weight_ne[3] && bcast_input_ne[3] == 1) {
        if (bcast_input_ne[2] == 1 && bcast_weight_ne[2] == 1) {
            n_dims = 2;
        } else if (bcast_input_ne[2] == 1) {
            n_dims = 3;
        }
    }

    acl_tensor_ptr acl_input_tensor = ggml_cann_create_tensor(input, bcast_input_ne, bcast_input_nb, n_dims);
    acl_tensor_ptr acl_weight_tensor;

    int64_t        transpose_ne[]   = { bcast_weight_ne[1], bcast_weight_ne[0], bcast_weight_ne[2],
                                        bcast_weight_ne[3], bcast_weight_ne[4], bcast_weight_ne[5] };
    size_t         transpose_nb[]   = { bcast_weight_nb[1], bcast_weight_nb[0], bcast_weight_nb[2],
                                        bcast_weight_nb[3], bcast_weight_nb[4], bcast_weight_nb[5] };

    aclFormat weight_format = ACL_FORMAT_ND;
    acl_weight_tensor = ggml_cann_create_tensor(weight, bcast_weight_ne, bcast_weight_nb, n_dims, weight_format);

    acl_tensor_ptr acl_dst = ggml_cann_create_tensor(dst, bcast_dst_ne, bcast_dst_nb, n_dims);

    // Cast if input is f32.
    acl_tensor_ptr       tmp_input_cast_tensor;
    ggml_cann_pool_alloc tmp_input_cast_allocator(ctx.pool());
    void *               tmp_input_cast_buffer = nullptr;

    if (input->type != GGML_TYPE_F16) {
        GGML_ASSERT(input->type == GGML_TYPE_F32);
        tmp_input_cast_allocator.alloc(ggml_nbytes(input));
        tmp_input_cast_buffer = tmp_input_cast_allocator.get();
        size_t * temp_input_cast_nb = new size_t[n_dims];
        temp_input_cast_nb[0] = ggml_type_size(GGML_TYPE_F16);
        for (int i = 1; i < n_dims; i++) {
            temp_input_cast_nb[i] = temp_input_cast_nb[i - 1] * bcast_input_ne[i - 1];
        }

        tmp_input_cast_tensor =
            ggml_cann_create_tensor(
                tmp_input_cast_buffer, ggml_cann_type_mapping(GGML_TYPE_F16), ggml_type_size(GGML_TYPE_F16),
                bcast_input_ne, temp_input_cast_nb, n_dims
            );
        aclnn_cast_custom(ctx, acl_input_tensor.get(), tmp_input_cast_tensor.get());
        GGML_LOG_INFO("Cast input to f16\n");
    }

    // Cast if weight is f32.
    acl_tensor_ptr       tmp_cast_tensor;
    ggml_cann_pool_alloc tmp_cast_allocator(ctx.pool());
    void *               tmp_cast_buffer = nullptr;
    if (weight->type != GGML_TYPE_F16) {
        GGML_ASSERT(weight->type == GGML_TYPE_F32);
        tmp_cast_allocator.alloc(ggml_nbytes(weight));
        tmp_cast_buffer = tmp_cast_allocator.get();
        size_t * temp_cast_nb = new size_t[n_dims];
        temp_cast_nb[0] = ggml_type_size(GGML_TYPE_F16);
        for (int i = 1; i < n_dims; i++) {
            temp_cast_nb[i] = temp_cast_nb[i - 1] * bcast_weight_ne[i - 1];
        }

        tmp_cast_tensor =
            ggml_cann_create_tensor(tmp_cast_buffer, ggml_cann_type_mapping(GGML_TYPE_F16), ggml_type_size(GGML_TYPE_F16),
                                    bcast_weight_ne, temp_cast_nb, n_dims, weight_format);
        aclnn_cast_custom(ctx, acl_weight_tensor.get(), tmp_cast_tensor.get());

        GGML_LOG_INFO("Cast weight to f16\n");
    }

    aclTensor * final_input_tensor = tmp_input_cast_tensor ? tmp_input_cast_tensor.get() : acl_input_tensor.get();
    aclTensor * final_weight_tensor = tmp_cast_tensor ? tmp_cast_tensor.get() : acl_weight_tensor.get();

    auto debug_acl_tensor = [](aclTensor * t, const char * name) {
        int64_t * dims = nullptr;
        uint64_t dim_num = 0;
        aclGetViewShape(t, &dims, &dim_num);

        int64_t * strides = nullptr;
        uint64_t stride_num = 0;
        aclGetViewStrides(t, &strides, &stride_num);

        aclDataType dtype;
        aclGetDataType(t, &dtype);

        aclFormat format;
        aclGetFormat(t, &format);

        printf("%s:\n", name);
        printf("  shape: (");
        for (uint64_t i = 0; i < dim_num; i++) {
            printf("%" PRId64 "%s", dims[i], i < dim_num - 1 ?  ", " : "");
        }
        printf(")\n  strides: (");
        for (uint64_t i = 0; i < stride_num; i++) {
            printf("%" PRId64 "%s", strides[i], i < stride_num - 1 ? ", " : "");
        }
        printf(")\n  dtype: %d, format: %d\n", (int)dtype, (int)format);
    };

    // debug_acl_tensor(final_input_tensor, "Input");
    // debug_acl_tensor(final_weight_tensor, "Weight");
    // debug_acl_tensor(acl_dst.get(), "Output");

    switch (n_dims) {
        case 2:
            GGML_CANN_CALL_ACLNN_OP(ctx, MmCustom, final_input_tensor, final_weight_tensor, false, true, acl_dst.get());
            break;
        case 3:
            GGML_CANN_CALL_ACLNN_OP(ctx, BatchMatmulCustom, final_input_tensor, final_weight_tensor, false, true, acl_dst.get());
            break;
        default:
            GGML_CANN_CALL_ACLNN_OP(ctx, MatmulCustom, final_input_tensor, final_weight_tensor, false, true, acl_dst.get());
            break;
    }
}

void ggml_cann_mul_mat_custom(ggml_backend_cann_context & ctx, ggml_tensor * dst) {
    const enum ggml_type type = dst->src[0]->type;
    switch (type) {
        case GGML_TYPE_F32:
        case GGML_TYPE_F16:
            ggml_cann_mat_mul_custom_fp(ctx, dst);
            break;
        case GGML_TYPE_Q4_0:
        case GGML_TYPE_Q8_0:
        default:
            GGML_ABORT("Unsupported type for mul_mat");
            break;
    }
}
