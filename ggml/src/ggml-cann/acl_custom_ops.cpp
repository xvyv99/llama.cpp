#include <cinttypes>
#include <cstdio>

#include <future>
#include <vector>
#include <cstring>

#include <cblas.h>

#include "acl_tensor.h"
#include "common.h"
#include "acl_custom_ops.h"

#include "catlass_kernel.h"

static void ggml_cann_mat_mul_custom_fp(ggml_backend_cann_context & ctx, ggml_tensor * dst) {
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

    const int64_t ne_plane      = ne01*ne00;
    const size_t  desired_wsize = type == GGML_TYPE_F32 ? 0 : ne03*ne02*ne_plane*sizeof(float);

    void * wdata = new char[desired_wsize];

    // convert src0 to float
    if (type != GGML_TYPE_F32) {
        const auto * type_traits = ggml_get_type_traits(type);
        ggml_to_float_t const to_float = type_traits->to_float;

        std::vector<std::future<void>> tasks;

        for (int64_t i03 = 0; i03 < ne03; i03++) {
            for (int64_t i02 = 0; i02 < ne02; i02++) {
                const void  *       x      = (char *)  src0->data + i02*nb02          + i03*nb03;
                        float * const wplane = (float *) wdata      + i02*ne_plane      + i03*ne02*ne_plane;

                const int min_cols_per_thread = 4096;
                const int min_rows_per_thread = std::max((int)(min_cols_per_thread/ne00), 1);
                const int n_threads = std::max(std::min(GGML_DEFAULT_N_THREADS, (int)(ne01/min_rows_per_thread)), 1);

                for (int i = 1; i < n_threads; i++) {
                    const int64_t start =       i*ne01/n_threads;
                    const int64_t end   = (i + 1)*ne01/n_threads;
                    if (start < end) {
                        tasks.push_back(std::async(std::launch::async, [=]() {
                            for (int64_t i01 = start; i01 < end; i01++) {
                                to_float((const char *) x + i01*nb01, wplane + i01*ne00, ne00);
                            }
                        }));
                    }
                }
                {
                    // reuse the current thread for the first task
                    const int64_t start = 0;
                    const int64_t end   = ne01/n_threads;
                    for (int64_t i01 = start; i01 < end; i01++) {
                        to_float((const char *) x + i01*nb01, wplane + i01*ne00, ne00);
                    }
                }
            }
        }

        // wait for all tasks to finish
        for (auto & task : tasks) {
            task.get();
        }
        tasks.clear();
    }

    for (int64_t i13 = 0; i13 < ne13; i13++) {
        for (int64_t i12 = 0; i12 < ne12; i12++) {
            const int64_t i03 = i13/r3;
            const int64_t i02 = i12/r2;

            const float * x = (float *) ((char *) src0->data + i02*nb02 + i03*nb03);
            const float * y = (float *) ((char *) src1->data + i12*nb12 + i13*nb13);
                    float * d = (float *) ((char *)  dst->data + i12*nb2  + i13*nb3);

            if (type != GGML_TYPE_F32) {
                x = (float *) wdata + i02*ne_plane + i03*ne02*ne_plane;
            }

            CatlassKernel::KernelInfo kernelInfo;
            kernelInfo.inputAddr = {
                reinterpret_cast<uint8_t *>(const_cast<float *>(y)),
                reinterpret_cast<uint8_t *>(const_cast<float *>(x))
            };
            kernelInfo.outputAddr = {reinterpret_cast<uint8_t *>(d)};
            kernelInfo.inputDataType = ACL_FLOAT;
            kernelInfo.outputDataType = ACL_FLOAT;
            kernelInfo.m = ne1;
            kernelInfo.n = ne01;
            kernelInfo.k = ne10;

            kernelInfo.lda = ne10;
            kernelInfo.ldb = ne00;
            kernelInfo.ldc = ne01;

            kernelInfo.transA = false;
            kernelInfo.transB = true;

            // CatlassKernel::BasicMatmul(1, ctx.stream(), kernelInfo);

            cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                        ne1, ne01, ne10,
                        1.0f,   y, ne10,
                                x, ne00,
                        0.0f,   d, ne01);

            // std::cout << "Shape: " << ne1 << "," << ne01 << "," << ne10 << std::endl;
        }
    }

    delete [] static_cast<char*>(wdata);
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
