#ifndef CANN_ACL_CUSTOM_OPS
#define CANN_ACL_CUSTOM_OPS

#include "acl/acl.h"
#include "aclnn/acl_meta.h"

#include "acl_tensor.h"
#include "common.h"

void ggml_cann_mul_mat_custom(ggml_backend_cann_context & ctx, ggml_tensor * dst);

#endif
