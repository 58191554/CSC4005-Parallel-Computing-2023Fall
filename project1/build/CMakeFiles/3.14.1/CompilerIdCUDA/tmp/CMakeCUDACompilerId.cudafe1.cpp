# 1 "CMakeCUDACompilerId.cu"
#pragma GCC diagnostic ignored "-Wunused-local-typedefs"
# 1
#pragma GCC diagnostic push
# 1
#pragma GCC diagnostic ignored "-Wunused-variable"
# 1
#pragma GCC diagnostic ignored "-Wunused-function"
# 1
static char __nv_inited_managed_rt = 0; static void **__nv_fatbinhandle_for_managed_rt; static void __nv_save_fatbinhandle_for_managed_rt(void **in){__nv_fatbinhandle_for_managed_rt = in;} static char __nv_init_managed_rt_with_module(void **); static inline void __nv_init_managed_rt(void) { __nv_inited_managed_rt = (__nv_inited_managed_rt ? __nv_inited_managed_rt                 : __nv_init_managed_rt_with_module(__nv_fatbinhandle_for_managed_rt));}
# 1
#pragma GCC diagnostic pop
# 1
#pragma GCC diagnostic ignored "-Wunused-variable"

# 1
#define __nv_is_extended_device_lambda_closure_type(X) false
#define __nv_is_extended_host_device_lambda_closure_type(X) false
#if defined(__nv_is_extended_device_lambda_closure_type) && defined(__nv_is_extended_host_device_lambda_closure_type)
#endif

# 1
# 61 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/cuda_runtime.h"
#pragma GCC diagnostic push
# 64
#pragma GCC diagnostic ignored "-Wunused-function"
# 68 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/device_types.h"
#if 0
# 68
enum cudaRoundMode { 
# 70
cudaRoundNearest, 
# 71
cudaRoundZero, 
# 72
cudaRoundPosInf, 
# 73
cudaRoundMinInf
# 74
}; 
#endif
# 100 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/vector_types.h"
#if 0
# 100
struct char1 { 
# 102
signed char x; 
# 103
}; 
#endif
# 105 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/vector_types.h"
#if 0
# 105
struct uchar1 { 
# 107
unsigned char x; 
# 108
}; 
#endif
# 111 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/vector_types.h"
#if 0
# 111
struct __attribute((aligned(2))) char2 { 
# 113
signed char x, y; 
# 114
}; 
#endif
# 116 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/vector_types.h"
#if 0
# 116
struct __attribute((aligned(2))) uchar2 { 
# 118
unsigned char x, y; 
# 119
}; 
#endif
# 121 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/vector_types.h"
#if 0
# 121
struct char3 { 
# 123
signed char x, y, z; 
# 124
}; 
#endif
# 126 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/vector_types.h"
#if 0
# 126
struct uchar3 { 
# 128
unsigned char x, y, z; 
# 129
}; 
#endif
# 131 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/vector_types.h"
#if 0
# 131
struct __attribute((aligned(4))) char4 { 
# 133
signed char x, y, z, w; 
# 134
}; 
#endif
# 136 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/vector_types.h"
#if 0
# 136
struct __attribute((aligned(4))) uchar4 { 
# 138
unsigned char x, y, z, w; 
# 139
}; 
#endif
# 141 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/vector_types.h"
#if 0
# 141
struct short1 { 
# 143
short x; 
# 144
}; 
#endif
# 146 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/vector_types.h"
#if 0
# 146
struct ushort1 { 
# 148
unsigned short x; 
# 149
}; 
#endif
# 151 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/vector_types.h"
#if 0
# 151
struct __attribute((aligned(4))) short2 { 
# 153
short x, y; 
# 154
}; 
#endif
# 156 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/vector_types.h"
#if 0
# 156
struct __attribute((aligned(4))) ushort2 { 
# 158
unsigned short x, y; 
# 159
}; 
#endif
# 161 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/vector_types.h"
#if 0
# 161
struct short3 { 
# 163
short x, y, z; 
# 164
}; 
#endif
# 166 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/vector_types.h"
#if 0
# 166
struct ushort3 { 
# 168
unsigned short x, y, z; 
# 169
}; 
#endif
# 171 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/vector_types.h"
#if 0
# 171
struct __attribute((aligned(8))) short4 { short x; short y; short z; short w; }; 
#endif
# 172 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/vector_types.h"
#if 0
# 172
struct __attribute((aligned(8))) ushort4 { unsigned short x; unsigned short y; unsigned short z; unsigned short w; }; 
#endif
# 174 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/vector_types.h"
#if 0
# 174
struct int1 { 
# 176
int x; 
# 177
}; 
#endif
# 179 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/vector_types.h"
#if 0
# 179
struct uint1 { 
# 181
unsigned x; 
# 182
}; 
#endif
# 184 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/vector_types.h"
#if 0
# 184
struct __attribute((aligned(8))) int2 { int x; int y; }; 
#endif
# 185 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/vector_types.h"
#if 0
# 185
struct __attribute((aligned(8))) uint2 { unsigned x; unsigned y; }; 
#endif
# 187 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/vector_types.h"
#if 0
# 187
struct int3 { 
# 189
int x, y, z; 
# 190
}; 
#endif
# 192 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/vector_types.h"
#if 0
# 192
struct uint3 { 
# 194
unsigned x, y, z; 
# 195
}; 
#endif
# 197 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/vector_types.h"
#if 0
# 197
struct __attribute((aligned(16))) int4 { 
# 199
int x, y, z, w; 
# 200
}; 
#endif
# 202 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/vector_types.h"
#if 0
# 202
struct __attribute((aligned(16))) uint4 { 
# 204
unsigned x, y, z, w; 
# 205
}; 
#endif
# 207 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/vector_types.h"
#if 0
# 207
struct long1 { 
# 209
long x; 
# 210
}; 
#endif
# 212 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/vector_types.h"
#if 0
# 212
struct ulong1 { 
# 214
unsigned long x; 
# 215
}; 
#endif
# 222 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/vector_types.h"
#if 0
# 222
struct __attribute((aligned((2) * sizeof(long)))) long2 { 
# 224
long x, y; 
# 225
}; 
#endif
# 227 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/vector_types.h"
#if 0
# 227
struct __attribute((aligned((2) * sizeof(unsigned long)))) ulong2 { 
# 229
unsigned long x, y; 
# 230
}; 
#endif
# 234 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/vector_types.h"
#if 0
# 234
struct long3 { 
# 236
long x, y, z; 
# 237
}; 
#endif
# 239 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/vector_types.h"
#if 0
# 239
struct ulong3 { 
# 241
unsigned long x, y, z; 
# 242
}; 
#endif
# 244 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/vector_types.h"
#if 0
# 244
struct __attribute((aligned(16))) long4 { 
# 246
long x, y, z, w; 
# 247
}; 
#endif
# 249 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/vector_types.h"
#if 0
# 249
struct __attribute((aligned(16))) ulong4 { 
# 251
unsigned long x, y, z, w; 
# 252
}; 
#endif
# 254 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/vector_types.h"
#if 0
# 254
struct float1 { 
# 256
float x; 
# 257
}; 
#endif
# 276 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/vector_types.h"
#if 0
# 276
struct __attribute((aligned(8))) float2 { float x; float y; }; 
#endif
# 281 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/vector_types.h"
#if 0
# 281
struct float3 { 
# 283
float x, y, z; 
# 284
}; 
#endif
# 286 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/vector_types.h"
#if 0
# 286
struct __attribute((aligned(16))) float4 { 
# 288
float x, y, z, w; 
# 289
}; 
#endif
# 291 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/vector_types.h"
#if 0
# 291
struct longlong1 { 
# 293
long long x; 
# 294
}; 
#endif
# 296 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/vector_types.h"
#if 0
# 296
struct ulonglong1 { 
# 298
unsigned long long x; 
# 299
}; 
#endif
# 301 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/vector_types.h"
#if 0
# 301
struct __attribute((aligned(16))) longlong2 { 
# 303
long long x, y; 
# 304
}; 
#endif
# 306 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/vector_types.h"
#if 0
# 306
struct __attribute((aligned(16))) ulonglong2 { 
# 308
unsigned long long x, y; 
# 309
}; 
#endif
# 311 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/vector_types.h"
#if 0
# 311
struct longlong3 { 
# 313
long long x, y, z; 
# 314
}; 
#endif
# 316 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/vector_types.h"
#if 0
# 316
struct ulonglong3 { 
# 318
unsigned long long x, y, z; 
# 319
}; 
#endif
# 321 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/vector_types.h"
#if 0
# 321
struct __attribute((aligned(16))) longlong4 { 
# 323
long long x, y, z, w; 
# 324
}; 
#endif
# 326 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/vector_types.h"
#if 0
# 326
struct __attribute((aligned(16))) ulonglong4 { 
# 328
unsigned long long x, y, z, w; 
# 329
}; 
#endif
# 331 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/vector_types.h"
#if 0
# 331
struct double1 { 
# 333
double x; 
# 334
}; 
#endif
# 336 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/vector_types.h"
#if 0
# 336
struct __attribute((aligned(16))) double2 { 
# 338
double x, y; 
# 339
}; 
#endif
# 341 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/vector_types.h"
#if 0
# 341
struct double3 { 
# 343
double x, y, z; 
# 344
}; 
#endif
# 346 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/vector_types.h"
#if 0
# 346
struct __attribute((aligned(16))) double4 { 
# 348
double x, y, z, w; 
# 349
}; 
#endif
# 363 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/vector_types.h"
#if 0
typedef char1 
# 363
char1; 
#endif
# 364 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/vector_types.h"
#if 0
typedef uchar1 
# 364
uchar1; 
#endif
# 365 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/vector_types.h"
#if 0
typedef char2 
# 365
char2; 
#endif
# 366 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/vector_types.h"
#if 0
typedef uchar2 
# 366
uchar2; 
#endif
# 367 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/vector_types.h"
#if 0
typedef char3 
# 367
char3; 
#endif
# 368 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/vector_types.h"
#if 0
typedef uchar3 
# 368
uchar3; 
#endif
# 369 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/vector_types.h"
#if 0
typedef char4 
# 369
char4; 
#endif
# 370 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/vector_types.h"
#if 0
typedef uchar4 
# 370
uchar4; 
#endif
# 371 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/vector_types.h"
#if 0
typedef short1 
# 371
short1; 
#endif
# 372 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/vector_types.h"
#if 0
typedef ushort1 
# 372
ushort1; 
#endif
# 373 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/vector_types.h"
#if 0
typedef short2 
# 373
short2; 
#endif
# 374 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/vector_types.h"
#if 0
typedef ushort2 
# 374
ushort2; 
#endif
# 375 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/vector_types.h"
#if 0
typedef short3 
# 375
short3; 
#endif
# 376 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/vector_types.h"
#if 0
typedef ushort3 
# 376
ushort3; 
#endif
# 377 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/vector_types.h"
#if 0
typedef short4 
# 377
short4; 
#endif
# 378 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/vector_types.h"
#if 0
typedef ushort4 
# 378
ushort4; 
#endif
# 379 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/vector_types.h"
#if 0
typedef int1 
# 379
int1; 
#endif
# 380 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/vector_types.h"
#if 0
typedef uint1 
# 380
uint1; 
#endif
# 381 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/vector_types.h"
#if 0
typedef int2 
# 381
int2; 
#endif
# 382 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/vector_types.h"
#if 0
typedef uint2 
# 382
uint2; 
#endif
# 383 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/vector_types.h"
#if 0
typedef int3 
# 383
int3; 
#endif
# 384 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/vector_types.h"
#if 0
typedef uint3 
# 384
uint3; 
#endif
# 385 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/vector_types.h"
#if 0
typedef int4 
# 385
int4; 
#endif
# 386 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/vector_types.h"
#if 0
typedef uint4 
# 386
uint4; 
#endif
# 387 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/vector_types.h"
#if 0
typedef long1 
# 387
long1; 
#endif
# 388 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/vector_types.h"
#if 0
typedef ulong1 
# 388
ulong1; 
#endif
# 389 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/vector_types.h"
#if 0
typedef long2 
# 389
long2; 
#endif
# 390 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/vector_types.h"
#if 0
typedef ulong2 
# 390
ulong2; 
#endif
# 391 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/vector_types.h"
#if 0
typedef long3 
# 391
long3; 
#endif
# 392 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/vector_types.h"
#if 0
typedef ulong3 
# 392
ulong3; 
#endif
# 393 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/vector_types.h"
#if 0
typedef long4 
# 393
long4; 
#endif
# 394 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/vector_types.h"
#if 0
typedef ulong4 
# 394
ulong4; 
#endif
# 395 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/vector_types.h"
#if 0
typedef float1 
# 395
float1; 
#endif
# 396 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/vector_types.h"
#if 0
typedef float2 
# 396
float2; 
#endif
# 397 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/vector_types.h"
#if 0
typedef float3 
# 397
float3; 
#endif
# 398 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/vector_types.h"
#if 0
typedef float4 
# 398
float4; 
#endif
# 399 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/vector_types.h"
#if 0
typedef longlong1 
# 399
longlong1; 
#endif
# 400 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/vector_types.h"
#if 0
typedef ulonglong1 
# 400
ulonglong1; 
#endif
# 401 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/vector_types.h"
#if 0
typedef longlong2 
# 401
longlong2; 
#endif
# 402 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/vector_types.h"
#if 0
typedef ulonglong2 
# 402
ulonglong2; 
#endif
# 403 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/vector_types.h"
#if 0
typedef longlong3 
# 403
longlong3; 
#endif
# 404 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/vector_types.h"
#if 0
typedef ulonglong3 
# 404
ulonglong3; 
#endif
# 405 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/vector_types.h"
#if 0
typedef longlong4 
# 405
longlong4; 
#endif
# 406 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/vector_types.h"
#if 0
typedef ulonglong4 
# 406
ulonglong4; 
#endif
# 407 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/vector_types.h"
#if 0
typedef double1 
# 407
double1; 
#endif
# 408 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/vector_types.h"
#if 0
typedef double2 
# 408
double2; 
#endif
# 409 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/vector_types.h"
#if 0
typedef double3 
# 409
double3; 
#endif
# 410 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/vector_types.h"
#if 0
typedef double4 
# 410
double4; 
#endif
# 418 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/vector_types.h"
#if 0
# 418
struct dim3 { 
# 420
unsigned x, y, z; 
# 432
}; 
#endif
# 434 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/vector_types.h"
#if 0
typedef dim3 
# 434
dim3; 
#endif
# 149 "/opt/rh/devtoolset-7/root/usr/lib/gcc/x86_64-redhat-linux/7/include/stddef.h" 3
typedef long ptrdiff_t; 
# 216 "/opt/rh/devtoolset-7/root/usr/lib/gcc/x86_64-redhat-linux/7/include/stddef.h" 3
typedef unsigned long size_t; 
#if !defined(__CUDA_INCLUDE_COMPILER_INTERNAL_HEADERS__)
#define __CUDA_INCLUDE_COMPILER_INTERNAL_HEADERS__
#endif
#include "crt/host_runtime.h"
# 437 "/opt/rh/devtoolset-7/root/usr/lib/gcc/x86_64-redhat-linux/7/include/stddef.h" 3
typedef 
# 426 "/opt/rh/devtoolset-7/root/usr/lib/gcc/x86_64-redhat-linux/7/include/stddef.h" 3
struct { 
# 427
long long __max_align_ll __attribute((__aligned__(__alignof__(long long)))); 
# 428
long double __max_align_ld __attribute((__aligned__(__alignof__(long double)))); 
# 437 "/opt/rh/devtoolset-7/root/usr/lib/gcc/x86_64-redhat-linux/7/include/stddef.h" 3
} max_align_t; 
# 444
typedef __decltype((nullptr)) nullptr_t; 
# 199 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/driver_types.h"
#if 0
# 199
enum cudaError { 
# 206
cudaSuccess, 
# 212
cudaErrorInvalidValue, 
# 218
cudaErrorMemoryAllocation, 
# 224
cudaErrorInitializationError, 
# 231
cudaErrorCudartUnloading, 
# 238
cudaErrorProfilerDisabled, 
# 246
cudaErrorProfilerNotInitialized, 
# 253
cudaErrorProfilerAlreadyStarted, 
# 260
cudaErrorProfilerAlreadyStopped, 
# 269 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/driver_types.h"
cudaErrorInvalidConfiguration, 
# 275
cudaErrorInvalidPitchValue = 12, 
# 281
cudaErrorInvalidSymbol, 
# 289
cudaErrorInvalidHostPointer = 16, 
# 297
cudaErrorInvalidDevicePointer, 
# 303
cudaErrorInvalidTexture, 
# 309
cudaErrorInvalidTextureBinding, 
# 316
cudaErrorInvalidChannelDescriptor, 
# 322
cudaErrorInvalidMemcpyDirection, 
# 332 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/driver_types.h"
cudaErrorAddressOfConstant, 
# 341 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/driver_types.h"
cudaErrorTextureFetchFailed, 
# 350 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/driver_types.h"
cudaErrorTextureNotBound, 
# 359 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/driver_types.h"
cudaErrorSynchronizationError, 
# 365
cudaErrorInvalidFilterSetting, 
# 371
cudaErrorInvalidNormSetting, 
# 379
cudaErrorMixedDeviceExecution, 
# 387
cudaErrorNotYetImplemented = 31, 
# 396 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/driver_types.h"
cudaErrorMemoryValueTooLarge, 
# 403
cudaErrorStubLibrary = 34, 
# 410
cudaErrorInsufficientDriver, 
# 417
cudaErrorCallRequiresNewerDriver, 
# 423
cudaErrorInvalidSurface, 
# 429
cudaErrorDuplicateVariableName = 43, 
# 435
cudaErrorDuplicateTextureName, 
# 441
cudaErrorDuplicateSurfaceName, 
# 451 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/driver_types.h"
cudaErrorDevicesUnavailable, 
# 464 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/driver_types.h"
cudaErrorIncompatibleDriverContext = 49, 
# 470
cudaErrorMissingConfiguration = 52, 
# 479 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/driver_types.h"
cudaErrorPriorLaunchFailure, 
# 486
cudaErrorLaunchMaxDepthExceeded = 65, 
# 494
cudaErrorLaunchFileScopedTex, 
# 502
cudaErrorLaunchFileScopedSurf, 
# 517 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/driver_types.h"
cudaErrorSyncDepthExceeded, 
# 529 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/driver_types.h"
cudaErrorLaunchPendingCountExceeded, 
# 535
cudaErrorInvalidDeviceFunction = 98, 
# 541
cudaErrorNoDevice = 100, 
# 548
cudaErrorInvalidDevice, 
# 553
cudaErrorDeviceNotLicensed, 
# 562 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/driver_types.h"
cudaErrorSoftwareValidityNotEstablished, 
# 567
cudaErrorStartupFailure = 127, 
# 572
cudaErrorInvalidKernelImage = 200, 
# 582 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/driver_types.h"
cudaErrorDeviceUninitialized, 
# 587
cudaErrorMapBufferObjectFailed = 205, 
# 592
cudaErrorUnmapBufferObjectFailed, 
# 598
cudaErrorArrayIsMapped, 
# 603
cudaErrorAlreadyMapped, 
# 611
cudaErrorNoKernelImageForDevice, 
# 616
cudaErrorAlreadyAcquired, 
# 621
cudaErrorNotMapped, 
# 627
cudaErrorNotMappedAsArray, 
# 633
cudaErrorNotMappedAsPointer, 
# 639
cudaErrorECCUncorrectable, 
# 645
cudaErrorUnsupportedLimit, 
# 651
cudaErrorDeviceAlreadyInUse, 
# 657
cudaErrorPeerAccessUnsupported, 
# 663
cudaErrorInvalidPtx, 
# 668
cudaErrorInvalidGraphicsContext, 
# 674
cudaErrorNvlinkUncorrectable, 
# 681
cudaErrorJitCompilerNotFound, 
# 688
cudaErrorUnsupportedPtxVersion, 
# 695
cudaErrorJitCompilationDisabled, 
# 700
cudaErrorUnsupportedExecAffinity, 
# 705
cudaErrorInvalidSource = 300, 
# 710
cudaErrorFileNotFound, 
# 715
cudaErrorSharedObjectSymbolNotFound, 
# 720
cudaErrorSharedObjectInitFailed, 
# 725
cudaErrorOperatingSystem, 
# 732
cudaErrorInvalidResourceHandle = 400, 
# 738
cudaErrorIllegalState, 
# 745
cudaErrorSymbolNotFound = 500, 
# 753
cudaErrorNotReady = 600, 
# 761
cudaErrorIllegalAddress = 700, 
# 770 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/driver_types.h"
cudaErrorLaunchOutOfResources, 
# 781 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/driver_types.h"
cudaErrorLaunchTimeout, 
# 787
cudaErrorLaunchIncompatibleTexturing, 
# 794
cudaErrorPeerAccessAlreadyEnabled, 
# 801
cudaErrorPeerAccessNotEnabled, 
# 814 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/driver_types.h"
cudaErrorSetOnActiveProcess = 708, 
# 821
cudaErrorContextIsDestroyed, 
# 828
cudaErrorAssert, 
# 835
cudaErrorTooManyPeers, 
# 841
cudaErrorHostMemoryAlreadyRegistered, 
# 847
cudaErrorHostMemoryNotRegistered, 
# 856 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/driver_types.h"
cudaErrorHardwareStackError, 
# 864
cudaErrorIllegalInstruction, 
# 873 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/driver_types.h"
cudaErrorMisalignedAddress, 
# 884 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/driver_types.h"
cudaErrorInvalidAddressSpace, 
# 892
cudaErrorInvalidPc, 
# 903 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/driver_types.h"
cudaErrorLaunchFailure, 
# 912 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/driver_types.h"
cudaErrorCooperativeLaunchTooLarge, 
# 917
cudaErrorNotPermitted = 800, 
# 923
cudaErrorNotSupported, 
# 932 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/driver_types.h"
cudaErrorSystemNotReady, 
# 939
cudaErrorSystemDriverMismatch, 
# 948 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/driver_types.h"
cudaErrorCompatNotSupportedOnDevice, 
# 953
cudaErrorMpsConnectionFailed, 
# 958
cudaErrorMpsRpcFailure, 
# 964
cudaErrorMpsServerNotReady, 
# 969
cudaErrorMpsMaxClientsReached, 
# 974
cudaErrorMpsMaxConnectionsReached, 
# 979
cudaErrorStreamCaptureUnsupported = 900, 
# 985
cudaErrorStreamCaptureInvalidated, 
# 991
cudaErrorStreamCaptureMerge, 
# 996
cudaErrorStreamCaptureUnmatched, 
# 1002
cudaErrorStreamCaptureUnjoined, 
# 1009
cudaErrorStreamCaptureIsolation, 
# 1015
cudaErrorStreamCaptureImplicit, 
# 1021
cudaErrorCapturedEvent, 
# 1028
cudaErrorStreamCaptureWrongThread, 
# 1033
cudaErrorTimeout, 
# 1039
cudaErrorGraphExecUpdateFailure, 
# 1044
cudaErrorUnknown = 999, 
# 1052
cudaErrorApiFailureBase = 10000
# 1053
}; 
#endif
# 1058 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/driver_types.h"
#if 0
# 1058
enum cudaChannelFormatKind { 
# 1060
cudaChannelFormatKindSigned, 
# 1061
cudaChannelFormatKindUnsigned, 
# 1062
cudaChannelFormatKindFloat, 
# 1063
cudaChannelFormatKindNone, 
# 1064
cudaChannelFormatKindNV12
# 1065
}; 
#endif
# 1070 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/driver_types.h"
#if 0
# 1070
struct cudaChannelFormatDesc { 
# 1072
int x; 
# 1073
int y; 
# 1074
int z; 
# 1075
int w; 
# 1076
cudaChannelFormatKind f; 
# 1077
}; 
#endif
# 1082 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/driver_types.h"
typedef struct cudaArray *cudaArray_t; 
# 1087
typedef const cudaArray *cudaArray_const_t; 
# 1089
struct cudaArray; 
# 1094
typedef struct cudaMipmappedArray *cudaMipmappedArray_t; 
# 1099
typedef const cudaMipmappedArray *cudaMipmappedArray_const_t; 
# 1101
struct cudaMipmappedArray; 
# 1111 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/driver_types.h"
#if 0
# 1111
struct cudaArraySparseProperties { 
# 1112
struct { 
# 1113
unsigned width; 
# 1114
unsigned height; 
# 1115
unsigned depth; 
# 1116
} tileExtent; 
# 1117
unsigned miptailFirstLevel; 
# 1118
unsigned long long miptailSize; 
# 1119
unsigned flags; 
# 1120
unsigned reserved[4]; 
# 1121
}; 
#endif
# 1126 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/driver_types.h"
#if 0
# 1126
enum cudaMemoryType { 
# 1128
cudaMemoryTypeUnregistered, 
# 1129
cudaMemoryTypeHost, 
# 1130
cudaMemoryTypeDevice, 
# 1131
cudaMemoryTypeManaged
# 1132
}; 
#endif
# 1137 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/driver_types.h"
#if 0
# 1137
enum cudaMemcpyKind { 
# 1139
cudaMemcpyHostToHost, 
# 1140
cudaMemcpyHostToDevice, 
# 1141
cudaMemcpyDeviceToHost, 
# 1142
cudaMemcpyDeviceToDevice, 
# 1143
cudaMemcpyDefault
# 1144
}; 
#endif
# 1151 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/driver_types.h"
#if 0
# 1151
struct cudaPitchedPtr { 
# 1153
void *ptr; 
# 1154
size_t pitch; 
# 1155
size_t xsize; 
# 1156
size_t ysize; 
# 1157
}; 
#endif
# 1164 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/driver_types.h"
#if 0
# 1164
struct cudaExtent { 
# 1166
size_t width; 
# 1167
size_t height; 
# 1168
size_t depth; 
# 1169
}; 
#endif
# 1176 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/driver_types.h"
#if 0
# 1176
struct cudaPos { 
# 1178
size_t x; 
# 1179
size_t y; 
# 1180
size_t z; 
# 1181
}; 
#endif
# 1186 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/driver_types.h"
#if 0
# 1186
struct cudaMemcpy3DParms { 
# 1188
cudaArray_t srcArray; 
# 1189
cudaPos srcPos; 
# 1190
cudaPitchedPtr srcPtr; 
# 1192
cudaArray_t dstArray; 
# 1193
cudaPos dstPos; 
# 1194
cudaPitchedPtr dstPtr; 
# 1196
cudaExtent extent; 
# 1197
cudaMemcpyKind kind; __pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)
# 1198
}; 
#endif
# 1203 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/driver_types.h"
#if 0
# 1203
struct cudaMemcpy3DPeerParms { 
# 1205
cudaArray_t srcArray; 
# 1206
cudaPos srcPos; 
# 1207
cudaPitchedPtr srcPtr; 
# 1208
int srcDevice; 
# 1210
cudaArray_t dstArray; 
# 1211
cudaPos dstPos; 
# 1212
cudaPitchedPtr dstPtr; 
# 1213
int dstDevice; 
# 1215
cudaExtent extent; 
# 1216
}; 
#endif
# 1221 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/driver_types.h"
#if 0
# 1221
struct cudaMemsetParams { 
# 1222
void *dst; 
# 1223
size_t pitch; 
# 1224
unsigned value; 
# 1225
unsigned elementSize; 
# 1226
size_t width; 
# 1227
size_t height; 
# 1228
}; 
#endif
# 1233 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/driver_types.h"
#if 0
# 1233
enum cudaAccessProperty { 
# 1234
cudaAccessPropertyNormal, 
# 1235
cudaAccessPropertyStreaming, 
# 1236
cudaAccessPropertyPersisting
# 1237
}; 
#endif
# 1250 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/driver_types.h"
#if 0
# 1250
struct cudaAccessPolicyWindow { 
# 1251
void *base_ptr; 
# 1252
size_t num_bytes; 
# 1253
float hitRatio; 
# 1254
cudaAccessProperty hitProp; 
# 1255
cudaAccessProperty missProp; __pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)
# 1256
}; 
#endif
# 1268 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/driver_types.h"
typedef void (*cudaHostFn_t)(void * userData); 
# 1273
#if 0
# 1273
struct cudaHostNodeParams { 
# 1274
cudaHostFn_t fn; 
# 1275
void *userData; 
# 1276
}; 
#endif
# 1281 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/driver_types.h"
#if 0
# 1281
enum cudaStreamCaptureStatus { 
# 1282
cudaStreamCaptureStatusNone, 
# 1283
cudaStreamCaptureStatusActive, 
# 1284
cudaStreamCaptureStatusInvalidated
# 1286
}; 
#endif
# 1292 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/driver_types.h"
#if 0
# 1292
enum cudaStreamCaptureMode { 
# 1293
cudaStreamCaptureModeGlobal, 
# 1294
cudaStreamCaptureModeThreadLocal, 
# 1295
cudaStreamCaptureModeRelaxed
# 1296
}; 
#endif
# 1298 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/driver_types.h"
#if 0
# 1298
enum cudaSynchronizationPolicy { 
# 1299
cudaSyncPolicyAuto = 1, 
# 1300
cudaSyncPolicySpin, 
# 1301
cudaSyncPolicyYield, 
# 1302
cudaSyncPolicyBlockingSync
# 1303
}; 
#endif
# 1308 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/driver_types.h"
#if 0
# 1308
enum cudaStreamAttrID { 
# 1309
cudaStreamAttributeAccessPolicyWindow = 1, 
# 1310
cudaStreamAttributeSynchronizationPolicy = 3
# 1311
}; 
#endif
# 1316 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/driver_types.h"
#if 0
# 1316
union cudaStreamAttrValue { 
# 1317
cudaAccessPolicyWindow accessPolicyWindow; 
# 1318
cudaSynchronizationPolicy syncPolicy; __pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)
# 1319
}; 
#endif
# 1324 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/driver_types.h"
#if 0
# 1324
enum cudaStreamUpdateCaptureDependenciesFlags { 
# 1325
cudaStreamAddCaptureDependencies, 
# 1326
cudaStreamSetCaptureDependencies
# 1327
}; 
#endif
# 1332 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/driver_types.h"
#if 0
# 1332
enum cudaUserObjectFlags { 
# 1333
cudaUserObjectNoDestructorSync = 1
# 1334
}; 
#endif
# 1339 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/driver_types.h"
#if 0
# 1339
enum cudaUserObjectRetainFlags { 
# 1340
cudaGraphUserObjectMove = 1
# 1341
}; 
#endif
# 1346 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/driver_types.h"
struct cudaGraphicsResource; 
# 1351
#if 0
# 1351
enum cudaGraphicsRegisterFlags { 
# 1353
cudaGraphicsRegisterFlagsNone, 
# 1354
cudaGraphicsRegisterFlagsReadOnly, 
# 1355
cudaGraphicsRegisterFlagsWriteDiscard, 
# 1356
cudaGraphicsRegisterFlagsSurfaceLoadStore = 4, 
# 1357
cudaGraphicsRegisterFlagsTextureGather = 8
# 1358
}; 
#endif
# 1363 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/driver_types.h"
#if 0
# 1363
enum cudaGraphicsMapFlags { 
# 1365
cudaGraphicsMapFlagsNone, 
# 1366
cudaGraphicsMapFlagsReadOnly, 
# 1367
cudaGraphicsMapFlagsWriteDiscard
# 1368
}; 
#endif
# 1373 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/driver_types.h"
#if 0
# 1373
enum cudaGraphicsCubeFace { 
# 1375
cudaGraphicsCubeFacePositiveX, 
# 1376
cudaGraphicsCubeFaceNegativeX, 
# 1377
cudaGraphicsCubeFacePositiveY, 
# 1378
cudaGraphicsCubeFaceNegativeY, 
# 1379
cudaGraphicsCubeFacePositiveZ, 
# 1380
cudaGraphicsCubeFaceNegativeZ
# 1381
}; 
#endif
# 1386 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/driver_types.h"
#if 0
# 1386
enum cudaKernelNodeAttrID { 
# 1387
cudaKernelNodeAttributeAccessPolicyWindow = 1, 
# 1388
cudaKernelNodeAttributeCooperative
# 1389
}; 
#endif
# 1394 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/driver_types.h"
#if 0
# 1394
union cudaKernelNodeAttrValue { 
# 1395
cudaAccessPolicyWindow accessPolicyWindow; 
# 1396
int cooperative; __pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)
# 1397
}; 
#endif
# 1402 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/driver_types.h"
#if 0
# 1402
enum cudaResourceType { 
# 1404
cudaResourceTypeArray, 
# 1405
cudaResourceTypeMipmappedArray, 
# 1406
cudaResourceTypeLinear, 
# 1407
cudaResourceTypePitch2D
# 1408
}; 
#endif
# 1413 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/driver_types.h"
#if 0
# 1413
enum cudaResourceViewFormat { 
# 1415
cudaResViewFormatNone, 
# 1416
cudaResViewFormatUnsignedChar1, 
# 1417
cudaResViewFormatUnsignedChar2, 
# 1418
cudaResViewFormatUnsignedChar4, 
# 1419
cudaResViewFormatSignedChar1, 
# 1420
cudaResViewFormatSignedChar2, 
# 1421
cudaResViewFormatSignedChar4, 
# 1422
cudaResViewFormatUnsignedShort1, 
# 1423
cudaResViewFormatUnsignedShort2, 
# 1424
cudaResViewFormatUnsignedShort4, 
# 1425
cudaResViewFormatSignedShort1, 
# 1426
cudaResViewFormatSignedShort2, 
# 1427
cudaResViewFormatSignedShort4, 
# 1428
cudaResViewFormatUnsignedInt1, 
# 1429
cudaResViewFormatUnsignedInt2, 
# 1430
cudaResViewFormatUnsignedInt4, 
# 1431
cudaResViewFormatSignedInt1, 
# 1432
cudaResViewFormatSignedInt2, 
# 1433
cudaResViewFormatSignedInt4, 
# 1434
cudaResViewFormatHalf1, 
# 1435
cudaResViewFormatHalf2, 
# 1436
cudaResViewFormatHalf4, 
# 1437
cudaResViewFormatFloat1, 
# 1438
cudaResViewFormatFloat2, 
# 1439
cudaResViewFormatFloat4, 
# 1440
cudaResViewFormatUnsignedBlockCompressed1, 
# 1441
cudaResViewFormatUnsignedBlockCompressed2, 
# 1442
cudaResViewFormatUnsignedBlockCompressed3, 
# 1443
cudaResViewFormatUnsignedBlockCompressed4, 
# 1444
cudaResViewFormatSignedBlockCompressed4, 
# 1445
cudaResViewFormatUnsignedBlockCompressed5, 
# 1446
cudaResViewFormatSignedBlockCompressed5, 
# 1447
cudaResViewFormatUnsignedBlockCompressed6H, 
# 1448
cudaResViewFormatSignedBlockCompressed6H, 
# 1449
cudaResViewFormatUnsignedBlockCompressed7
# 1450
}; 
#endif
# 1455 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/driver_types.h"
#if 0
# 1455
struct cudaResourceDesc { 
# 1456
cudaResourceType resType; 
# 1458
union { 
# 1459
struct { 
# 1460
cudaArray_t array; 
# 1461
} array; 
# 1462
struct { 
# 1463
cudaMipmappedArray_t mipmap; 
# 1464
} mipmap; 
# 1465
struct { 
# 1466
void *devPtr; 
# 1467
cudaChannelFormatDesc desc; 
# 1468
size_t sizeInBytes; 
# 1469
} linear; 
# 1470
struct { 
# 1471
void *devPtr; 
# 1472
cudaChannelFormatDesc desc; 
# 1473
size_t width; 
# 1474
size_t height; 
# 1475
size_t pitchInBytes; 
# 1476
} pitch2D; 
# 1477
} res; 
# 1478
}; 
#endif
# 1483 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/driver_types.h"
#if 0
# 1483
struct cudaResourceViewDesc { 
# 1485
cudaResourceViewFormat format; 
# 1486
size_t width; 
# 1487
size_t height; 
# 1488
size_t depth; 
# 1489
unsigned firstMipmapLevel; 
# 1490
unsigned lastMipmapLevel; 
# 1491
unsigned firstLayer; 
# 1492
unsigned lastLayer; 
# 1493
}; 
#endif
# 1498 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/driver_types.h"
#if 0
# 1498
struct cudaPointerAttributes { 
# 1504
cudaMemoryType type; 
# 1515 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/driver_types.h"
int device; 
# 1521
void *devicePointer; 
# 1530 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/driver_types.h"
void *hostPointer; 
# 1531
}; 
#endif
# 1536 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/driver_types.h"
#if 0
# 1536
struct cudaFuncAttributes { 
# 1543
size_t sharedSizeBytes; 
# 1549
size_t constSizeBytes; 
# 1554
size_t localSizeBytes; 
# 1561
int maxThreadsPerBlock; 
# 1566
int numRegs; 
# 1573
int ptxVersion; 
# 1580
int binaryVersion; 
# 1586
int cacheModeCA; 
# 1593
int maxDynamicSharedSizeBytes; 
# 1602 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/driver_types.h"
int preferredShmemCarveout; 
# 1603
}; 
#endif
# 1608 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/driver_types.h"
#if 0
# 1608
enum cudaFuncAttribute { 
# 1610
cudaFuncAttributeMaxDynamicSharedMemorySize = 8, 
# 1611
cudaFuncAttributePreferredSharedMemoryCarveout, 
# 1612
cudaFuncAttributeMax
# 1613
}; 
#endif
# 1618 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/driver_types.h"
#if 0
# 1618
enum cudaFuncCache { 
# 1620
cudaFuncCachePreferNone, 
# 1621
cudaFuncCachePreferShared, 
# 1622
cudaFuncCachePreferL1, 
# 1623
cudaFuncCachePreferEqual
# 1624
}; 
#endif
# 1630 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/driver_types.h"
#if 0
# 1630
enum cudaSharedMemConfig { 
# 1632
cudaSharedMemBankSizeDefault, 
# 1633
cudaSharedMemBankSizeFourByte, 
# 1634
cudaSharedMemBankSizeEightByte
# 1635
}; 
#endif
# 1640 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/driver_types.h"
#if 0
# 1640
enum cudaSharedCarveout { 
# 1641
cudaSharedmemCarveoutDefault = (-1), 
# 1642
cudaSharedmemCarveoutMaxShared = 100, 
# 1643
cudaSharedmemCarveoutMaxL1 = 0
# 1644
}; 
#endif
# 1649 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/driver_types.h"
#if 0
# 1649
enum cudaComputeMode { 
# 1651
cudaComputeModeDefault, 
# 1652
cudaComputeModeExclusive, 
# 1653
cudaComputeModeProhibited, 
# 1654
cudaComputeModeExclusiveProcess
# 1655
}; 
#endif
# 1660 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/driver_types.h"
#if 0
# 1660
enum cudaLimit { 
# 1662
cudaLimitStackSize, 
# 1663
cudaLimitPrintfFifoSize, 
# 1664
cudaLimitMallocHeapSize, 
# 1665
cudaLimitDevRuntimeSyncDepth, 
# 1666
cudaLimitDevRuntimePendingLaunchCount, 
# 1667
cudaLimitMaxL2FetchGranularity, 
# 1668
cudaLimitPersistingL2CacheSize
# 1669
}; 
#endif
# 1674 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/driver_types.h"
#if 0
# 1674
enum cudaMemoryAdvise { 
# 1676
cudaMemAdviseSetReadMostly = 1, 
# 1677
cudaMemAdviseUnsetReadMostly, 
# 1678
cudaMemAdviseSetPreferredLocation, 
# 1679
cudaMemAdviseUnsetPreferredLocation, 
# 1680
cudaMemAdviseSetAccessedBy, 
# 1681
cudaMemAdviseUnsetAccessedBy
# 1682
}; 
#endif
# 1687 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/driver_types.h"
#if 0
# 1687
enum cudaMemRangeAttribute { 
# 1689
cudaMemRangeAttributeReadMostly = 1, 
# 1690
cudaMemRangeAttributePreferredLocation, 
# 1691
cudaMemRangeAttributeAccessedBy, 
# 1692
cudaMemRangeAttributeLastPrefetchLocation
# 1693
}; 
#endif
# 1698 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/driver_types.h"
#if 0
# 1698
enum cudaOutputMode { 
# 1700
cudaKeyValuePair, 
# 1701
cudaCSV
# 1702
}; 
#endif
# 1707 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/driver_types.h"
#if 0
# 1707
enum cudaFlushGPUDirectRDMAWritesOptions { 
# 1708
cudaFlushGPUDirectRDMAWritesOptionHost = (1 << 0), 
# 1709
cudaFlushGPUDirectRDMAWritesOptionMemOps
# 1710
}; 
#endif
# 1715 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/driver_types.h"
#if 0
# 1715
enum cudaGPUDirectRDMAWritesOrdering { 
# 1716
cudaGPUDirectRDMAWritesOrderingNone, 
# 1717
cudaGPUDirectRDMAWritesOrderingOwner = 100, 
# 1718
cudaGPUDirectRDMAWritesOrderingAllDevices = 200
# 1719
}; 
#endif
# 1724 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/driver_types.h"
#if 0
# 1724
enum cudaFlushGPUDirectRDMAWritesScope { 
# 1725
cudaFlushGPUDirectRDMAWritesToOwner = 100, 
# 1726
cudaFlushGPUDirectRDMAWritesToAllDevices = 200
# 1727
}; 
#endif
# 1732 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/driver_types.h"
#if 0
# 1732
enum cudaFlushGPUDirectRDMAWritesTarget { 
# 1733
cudaFlushGPUDirectRDMAWritesTargetCurrentDevice
# 1734
}; 
#endif
# 1740 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/driver_types.h"
#if 0
# 1740
enum cudaDeviceAttr { 
# 1742
cudaDevAttrMaxThreadsPerBlock = 1, 
# 1743
cudaDevAttrMaxBlockDimX, 
# 1744
cudaDevAttrMaxBlockDimY, 
# 1745
cudaDevAttrMaxBlockDimZ, 
# 1746
cudaDevAttrMaxGridDimX, 
# 1747
cudaDevAttrMaxGridDimY, 
# 1748
cudaDevAttrMaxGridDimZ, 
# 1749
cudaDevAttrMaxSharedMemoryPerBlock, 
# 1750
cudaDevAttrTotalConstantMemory, 
# 1751
cudaDevAttrWarpSize, 
# 1752
cudaDevAttrMaxPitch, 
# 1753
cudaDevAttrMaxRegistersPerBlock, 
# 1754
cudaDevAttrClockRate, 
# 1755
cudaDevAttrTextureAlignment, 
# 1756
cudaDevAttrGpuOverlap, 
# 1757
cudaDevAttrMultiProcessorCount, 
# 1758
cudaDevAttrKernelExecTimeout, 
# 1759
cudaDevAttrIntegrated, 
# 1760
cudaDevAttrCanMapHostMemory, 
# 1761
cudaDevAttrComputeMode, 
# 1762
cudaDevAttrMaxTexture1DWidth, 
# 1763
cudaDevAttrMaxTexture2DWidth, 
# 1764
cudaDevAttrMaxTexture2DHeight, 
# 1765
cudaDevAttrMaxTexture3DWidth, 
# 1766
cudaDevAttrMaxTexture3DHeight, 
# 1767
cudaDevAttrMaxTexture3DDepth, 
# 1768
cudaDevAttrMaxTexture2DLayeredWidth, 
# 1769
cudaDevAttrMaxTexture2DLayeredHeight, 
# 1770
cudaDevAttrMaxTexture2DLayeredLayers, 
# 1771
cudaDevAttrSurfaceAlignment, 
# 1772
cudaDevAttrConcurrentKernels, 
# 1773
cudaDevAttrEccEnabled, 
# 1774
cudaDevAttrPciBusId, 
# 1775
cudaDevAttrPciDeviceId, 
# 1776
cudaDevAttrTccDriver, 
# 1777
cudaDevAttrMemoryClockRate, 
# 1778
cudaDevAttrGlobalMemoryBusWidth, 
# 1779
cudaDevAttrL2CacheSize, 
# 1780
cudaDevAttrMaxThreadsPerMultiProcessor, 
# 1781
cudaDevAttrAsyncEngineCount, 
# 1782
cudaDevAttrUnifiedAddressing, 
# 1783
cudaDevAttrMaxTexture1DLayeredWidth, 
# 1784
cudaDevAttrMaxTexture1DLayeredLayers, 
# 1785
cudaDevAttrMaxTexture2DGatherWidth = 45, 
# 1786
cudaDevAttrMaxTexture2DGatherHeight, 
# 1787
cudaDevAttrMaxTexture3DWidthAlt, 
# 1788
cudaDevAttrMaxTexture3DHeightAlt, 
# 1789
cudaDevAttrMaxTexture3DDepthAlt, 
# 1790
cudaDevAttrPciDomainId, 
# 1791
cudaDevAttrTexturePitchAlignment, 
# 1792
cudaDevAttrMaxTextureCubemapWidth, 
# 1793
cudaDevAttrMaxTextureCubemapLayeredWidth, 
# 1794
cudaDevAttrMaxTextureCubemapLayeredLayers, 
# 1795
cudaDevAttrMaxSurface1DWidth, 
# 1796
cudaDevAttrMaxSurface2DWidth, 
# 1797
cudaDevAttrMaxSurface2DHeight, 
# 1798
cudaDevAttrMaxSurface3DWidth, 
# 1799
cudaDevAttrMaxSurface3DHeight, 
# 1800
cudaDevAttrMaxSurface3DDepth, 
# 1801
cudaDevAttrMaxSurface1DLayeredWidth, 
# 1802
cudaDevAttrMaxSurface1DLayeredLayers, 
# 1803
cudaDevAttrMaxSurface2DLayeredWidth, 
# 1804
cudaDevAttrMaxSurface2DLayeredHeight, 
# 1805
cudaDevAttrMaxSurface2DLayeredLayers, 
# 1806
cudaDevAttrMaxSurfaceCubemapWidth, 
# 1807
cudaDevAttrMaxSurfaceCubemapLayeredWidth, 
# 1808
cudaDevAttrMaxSurfaceCubemapLayeredLayers, 
# 1809
cudaDevAttrMaxTexture1DLinearWidth, 
# 1810
cudaDevAttrMaxTexture2DLinearWidth, 
# 1811
cudaDevAttrMaxTexture2DLinearHeight, 
# 1812
cudaDevAttrMaxTexture2DLinearPitch, 
# 1813
cudaDevAttrMaxTexture2DMipmappedWidth, 
# 1814
cudaDevAttrMaxTexture2DMipmappedHeight, 
# 1815
cudaDevAttrComputeCapabilityMajor, 
# 1816
cudaDevAttrComputeCapabilityMinor, 
# 1817
cudaDevAttrMaxTexture1DMipmappedWidth, 
# 1818
cudaDevAttrStreamPrioritiesSupported, 
# 1819
cudaDevAttrGlobalL1CacheSupported, 
# 1820
cudaDevAttrLocalL1CacheSupported, 
# 1821
cudaDevAttrMaxSharedMemoryPerMultiprocessor, 
# 1822
cudaDevAttrMaxRegistersPerMultiprocessor, 
# 1823
cudaDevAttrManagedMemory, 
# 1824
cudaDevAttrIsMultiGpuBoard, 
# 1825
cudaDevAttrMultiGpuBoardGroupID, 
# 1826
cudaDevAttrHostNativeAtomicSupported, 
# 1827
cudaDevAttrSingleToDoublePrecisionPerfRatio, 
# 1828
cudaDevAttrPageableMemoryAccess, 
# 1829
cudaDevAttrConcurrentManagedAccess, 
# 1830
cudaDevAttrComputePreemptionSupported, 
# 1831
cudaDevAttrCanUseHostPointerForRegisteredMem, 
# 1832
cudaDevAttrReserved92, 
# 1833
cudaDevAttrReserved93, 
# 1834
cudaDevAttrReserved94, 
# 1835
cudaDevAttrCooperativeLaunch, 
# 1836
cudaDevAttrCooperativeMultiDeviceLaunch, 
# 1837
cudaDevAttrMaxSharedMemoryPerBlockOptin, 
# 1838
cudaDevAttrCanFlushRemoteWrites, 
# 1839
cudaDevAttrHostRegisterSupported, 
# 1840
cudaDevAttrPageableMemoryAccessUsesHostPageTables, 
# 1841
cudaDevAttrDirectManagedMemAccessFromHost, 
# 1842
cudaDevAttrMaxBlocksPerMultiprocessor = 106, 
# 1843
cudaDevAttrMaxPersistingL2CacheSize = 108, 
# 1844
cudaDevAttrMaxAccessPolicyWindowSize, 
# 1845
cudaDevAttrReservedSharedMemoryPerBlock = 111, 
# 1846
cudaDevAttrSparseCudaArraySupported, 
# 1847
cudaDevAttrHostRegisterReadOnlySupported, 
# 1848
cudaDevAttrMaxTimelineSemaphoreInteropSupported, 
# 1849
cudaDevAttrMemoryPoolsSupported, 
# 1850
cudaDevAttrGPUDirectRDMASupported, 
# 1851
cudaDevAttrGPUDirectRDMAFlushWritesOptions, 
# 1852
cudaDevAttrGPUDirectRDMAWritesOrdering, 
# 1853
cudaDevAttrMemoryPoolSupportedHandleTypes, 
# 1854
cudaDevAttrMax
# 1855
}; 
#endif
# 1860 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/driver_types.h"
#if 0
# 1860
enum cudaMemPoolAttr { 
# 1870 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/driver_types.h"
cudaMemPoolReuseFollowEventDependencies = 1, 
# 1877
cudaMemPoolReuseAllowOpportunistic, 
# 1885
cudaMemPoolReuseAllowInternalDependencies, 
# 1896 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/driver_types.h"
cudaMemPoolAttrReleaseThreshold, 
# 1902
cudaMemPoolAttrReservedMemCurrent, 
# 1909
cudaMemPoolAttrReservedMemHigh, 
# 1915
cudaMemPoolAttrUsedMemCurrent, 
# 1922
cudaMemPoolAttrUsedMemHigh
# 1923
}; 
#endif
# 1928 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/driver_types.h"
#if 0
# 1928
enum cudaMemLocationType { 
# 1929
cudaMemLocationTypeInvalid, 
# 1930
cudaMemLocationTypeDevice
# 1931
}; 
#endif
# 1938 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/driver_types.h"
#if 0
# 1938
struct cudaMemLocation { 
# 1939
cudaMemLocationType type; 
# 1940
int id; 
# 1941
}; 
#endif
# 1946 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/driver_types.h"
#if 0
# 1946
enum cudaMemAccessFlags { 
# 1947
cudaMemAccessFlagsProtNone, 
# 1948
cudaMemAccessFlagsProtRead, 
# 1949
cudaMemAccessFlagsProtReadWrite = 3
# 1950
}; 
#endif
# 1955 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/driver_types.h"
#if 0
# 1955
struct cudaMemAccessDesc { 
# 1956
cudaMemLocation location; 
# 1957
cudaMemAccessFlags flags; 
# 1958
}; 
#endif
# 1963 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/driver_types.h"
#if 0
# 1963
enum cudaMemAllocationType { 
# 1964
cudaMemAllocationTypeInvalid, 
# 1968
cudaMemAllocationTypePinned, 
# 1969
cudaMemAllocationTypeMax = 2147483647
# 1970
}; 
#endif
# 1975 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/driver_types.h"
#if 0
# 1975
enum cudaMemAllocationHandleType { 
# 1976
cudaMemHandleTypeNone, 
# 1977
cudaMemHandleTypePosixFileDescriptor, 
# 1978
cudaMemHandleTypeWin32, 
# 1979
cudaMemHandleTypeWin32Kmt = 4
# 1980
}; 
#endif
# 1985 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/driver_types.h"
#if 0
# 1985
struct cudaMemPoolProps { 
# 1986
cudaMemAllocationType allocType; 
# 1987
cudaMemAllocationHandleType handleTypes; 
# 1988
cudaMemLocation location; 
# 1995
void *win32SecurityAttributes; 
# 1996
unsigned char reserved[64]; 
# 1997
}; 
#endif
# 2002 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/driver_types.h"
#if 0
# 2002
struct cudaMemPoolPtrExportData { 
# 2003
unsigned char reserved[64]; 
# 2004
}; 
#endif
# 2009 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/driver_types.h"
#if 0
# 2009
struct cudaMemAllocNodeParams { 
# 2014
cudaMemPoolProps poolProps; 
# 2015
const cudaMemAccessDesc *accessDescs; 
# 2016
size_t accessDescCount; 
# 2017
size_t bytesize; 
# 2018
void *dptr; 
# 2019
}; 
#endif
# 2024 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/driver_types.h"
#if 0
# 2024
enum cudaGraphMemAttributeType { 
# 2029
cudaGraphMemAttrUsedMemCurrent = 1, 
# 2036
cudaGraphMemAttrUsedMemHigh, 
# 2043
cudaGraphMemAttrReservedMemCurrent, 
# 2050
cudaGraphMemAttrReservedMemHigh
# 2051
}; 
#endif
# 2057 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/driver_types.h"
#if 0
# 2057
enum cudaDeviceP2PAttr { 
# 2058
cudaDevP2PAttrPerformanceRank = 1, 
# 2059
cudaDevP2PAttrAccessSupported, 
# 2060
cudaDevP2PAttrNativeAtomicSupported, 
# 2061
cudaDevP2PAttrCudaArrayAccessSupported
# 2062
}; 
#endif
# 2069 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/driver_types.h"
#if 0
# 2069
struct CUuuid_st { 
# 2070
char bytes[16]; 
# 2071
}; 
#endif
# 2072 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/driver_types.h"
#if 0
typedef CUuuid_st 
# 2072
CUuuid; 
#endif
# 2074 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/driver_types.h"
#if 0
typedef CUuuid_st 
# 2074
cudaUUID_t; 
#endif
# 2079 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/driver_types.h"
#if 0
# 2079
struct cudaDeviceProp { 
# 2081
char name[256]; 
# 2082
cudaUUID_t uuid; 
# 2083
char luid[8]; 
# 2084
unsigned luidDeviceNodeMask; 
# 2085
size_t totalGlobalMem; 
# 2086
size_t sharedMemPerBlock; 
# 2087
int regsPerBlock; 
# 2088
int warpSize; 
# 2089
size_t memPitch; 
# 2090
int maxThreadsPerBlock; 
# 2091
int maxThreadsDim[3]; 
# 2092
int maxGridSize[3]; 
# 2093
int clockRate; 
# 2094
size_t totalConstMem; 
# 2095
int major; 
# 2096
int minor; 
# 2097
size_t textureAlignment; 
# 2098
size_t texturePitchAlignment; 
# 2099
int deviceOverlap; 
# 2100
int multiProcessorCount; 
# 2101
int kernelExecTimeoutEnabled; 
# 2102
int integrated; 
# 2103
int canMapHostMemory; 
# 2104
int computeMode; 
# 2105
int maxTexture1D; 
# 2106
int maxTexture1DMipmap; 
# 2107
int maxTexture1DLinear; 
# 2108
int maxTexture2D[2]; 
# 2109
int maxTexture2DMipmap[2]; 
# 2110
int maxTexture2DLinear[3]; 
# 2111
int maxTexture2DGather[2]; 
# 2112
int maxTexture3D[3]; 
# 2113
int maxTexture3DAlt[3]; 
# 2114
int maxTextureCubemap; 
# 2115
int maxTexture1DLayered[2]; 
# 2116
int maxTexture2DLayered[3]; 
# 2117
int maxTextureCubemapLayered[2]; 
# 2118
int maxSurface1D; 
# 2119
int maxSurface2D[2]; 
# 2120
int maxSurface3D[3]; 
# 2121
int maxSurface1DLayered[2]; 
# 2122
int maxSurface2DLayered[3]; 
# 2123
int maxSurfaceCubemap; 
# 2124
int maxSurfaceCubemapLayered[2]; 
# 2125
size_t surfaceAlignment; 
# 2126
int concurrentKernels; 
# 2127
int ECCEnabled; 
# 2128
int pciBusID; 
# 2129
int pciDeviceID; 
# 2130
int pciDomainID; 
# 2131
int tccDriver; 
# 2132
int asyncEngineCount; 
# 2133
int unifiedAddressing; 
# 2134
int memoryClockRate; 
# 2135
int memoryBusWidth; 
# 2136
int l2CacheSize; 
# 2137
int persistingL2CacheMaxSize; 
# 2138
int maxThreadsPerMultiProcessor; 
# 2139
int streamPrioritiesSupported; 
# 2140
int globalL1CacheSupported; 
# 2141
int localL1CacheSupported; 
# 2142
size_t sharedMemPerMultiprocessor; 
# 2143
int regsPerMultiprocessor; 
# 2144
int managedMemory; 
# 2145
int isMultiGpuBoard; 
# 2146
int multiGpuBoardGroupID; 
# 2147
int hostNativeAtomicSupported; 
# 2148
int singleToDoublePrecisionPerfRatio; 
# 2149
int pageableMemoryAccess; 
# 2150
int concurrentManagedAccess; 
# 2151
int computePreemptionSupported; 
# 2152
int canUseHostPointerForRegisteredMem; 
# 2153
int cooperativeLaunch; 
# 2154
int cooperativeMultiDeviceLaunch; 
# 2155
size_t sharedMemPerBlockOptin; 
# 2156
int pageableMemoryAccessUsesHostPageTables; 
# 2157
int directManagedMemAccessFromHost; 
# 2158
int maxBlocksPerMultiProcessor; 
# 2159
int accessPolicyMaxWindowSize; 
# 2160
size_t reservedSharedMemPerBlock; 
# 2161
}; 
#endif
# 2257 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/driver_types.h"
#if 0
typedef 
# 2254
struct cudaIpcEventHandle_st { 
# 2256
char reserved[64]; 
# 2257
} cudaIpcEventHandle_t; 
#endif
# 2265 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/driver_types.h"
#if 0
typedef 
# 2262
struct cudaIpcMemHandle_st { 
# 2264
char reserved[64]; 
# 2265
} cudaIpcMemHandle_t; 
#endif
# 2270 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/driver_types.h"
#if 0
# 2270
enum cudaExternalMemoryHandleType { 
# 2274
cudaExternalMemoryHandleTypeOpaqueFd = 1, 
# 2278
cudaExternalMemoryHandleTypeOpaqueWin32, 
# 2282
cudaExternalMemoryHandleTypeOpaqueWin32Kmt, 
# 2286
cudaExternalMemoryHandleTypeD3D12Heap, 
# 2290
cudaExternalMemoryHandleTypeD3D12Resource, 
# 2294
cudaExternalMemoryHandleTypeD3D11Resource, 
# 2298
cudaExternalMemoryHandleTypeD3D11ResourceKmt, 
# 2302
cudaExternalMemoryHandleTypeNvSciBuf
# 2303
}; 
#endif
# 2345 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/driver_types.h"
#if 0
# 2345
struct cudaExternalMemoryHandleDesc { 
# 2349
cudaExternalMemoryHandleType type; 
# 2350
union { 
# 2356
int fd; 
# 2372 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/driver_types.h"
struct { 
# 2376
void *handle; 
# 2381
const void *name; 
# 2382
} win32; 
# 2387
const void *nvSciBufObject; 
# 2388
} handle; 
# 2392
unsigned long long size; 
# 2396
unsigned flags; __pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)
# 2397
}; 
#endif
# 2402 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/driver_types.h"
#if 0
# 2402
struct cudaExternalMemoryBufferDesc { 
# 2406
unsigned long long offset; 
# 2410
unsigned long long size; 
# 2414
unsigned flags; 
# 2415
}; 
#endif
# 2420 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/driver_types.h"
#if 0
# 2420
struct cudaExternalMemoryMipmappedArrayDesc { 
# 2425
unsigned long long offset; 
# 2429
cudaChannelFormatDesc formatDesc; 
# 2433
cudaExtent extent; 
# 2438
unsigned flags; 
# 2442
unsigned numLevels; 
# 2443
}; 
#endif
# 2448 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/driver_types.h"
#if 0
# 2448
enum cudaExternalSemaphoreHandleType { 
# 2452
cudaExternalSemaphoreHandleTypeOpaqueFd = 1, 
# 2456
cudaExternalSemaphoreHandleTypeOpaqueWin32, 
# 2460
cudaExternalSemaphoreHandleTypeOpaqueWin32Kmt, 
# 2464
cudaExternalSemaphoreHandleTypeD3D12Fence, 
# 2468
cudaExternalSemaphoreHandleTypeD3D11Fence, 
# 2472
cudaExternalSemaphoreHandleTypeNvSciSync, 
# 2476
cudaExternalSemaphoreHandleTypeKeyedMutex, 
# 2480
cudaExternalSemaphoreHandleTypeKeyedMutexKmt, 
# 2484
cudaExternalSemaphoreHandleTypeTimelineSemaphoreFd, 
# 2488
cudaExternalSemaphoreHandleTypeTimelineSemaphoreWin32
# 2489
}; 
#endif
# 2494 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/driver_types.h"
#if 0
# 2494
struct cudaExternalSemaphoreHandleDesc { 
# 2498
cudaExternalSemaphoreHandleType type; 
# 2499
union { 
# 2506
int fd; 
# 2522 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/driver_types.h"
struct { 
# 2526
void *handle; 
# 2531
const void *name; 
# 2532
} win32; 
# 2536
const void *nvSciSyncObj; 
# 2537
} handle; 
# 2541
unsigned flags; __pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)
# 2542
}; 
#endif
# 2643 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/driver_types.h"
#if 0
# 2643
struct cudaExternalSemaphoreSignalParams { 
# 2644
struct { 
# 2648
struct { 
# 2652
unsigned long long value; 
# 2653
} fence; 
# 2654
union { 
# 2659
void *fence; 
# 2660
unsigned long long reserved; 
# 2661
} nvSciSync; 
# 2665
struct { 
# 2669
unsigned long long key; 
# 2670
} keyedMutex; 
# 2671
unsigned reserved[12]; 
# 2672
} params; 
# 2683 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/driver_types.h"
unsigned flags; 
# 2684
unsigned reserved[16]; 
# 2685
}; 
#endif
# 2690 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/driver_types.h"
#if 0
# 2690
struct cudaExternalSemaphoreWaitParams { 
# 2691
struct { 
# 2695
struct { 
# 2699
unsigned long long value; 
# 2700
} fence; 
# 2701
union { 
# 2706
void *fence; 
# 2707
unsigned long long reserved; 
# 2708
} nvSciSync; 
# 2712
struct { 
# 2716
unsigned long long key; 
# 2720
unsigned timeoutMs; 
# 2721
} keyedMutex; 
# 2722
unsigned reserved[10]; 
# 2723
} params; 
# 2734 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/driver_types.h"
unsigned flags; 
# 2735
unsigned reserved[16]; 
# 2736
}; 
#endif
# 2748 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/driver_types.h"
#if 0
typedef cudaError 
# 2748
cudaError_t; 
#endif
# 2753 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/driver_types.h"
#if 0
typedef struct CUstream_st *
# 2753
cudaStream_t; 
#endif
# 2758 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/driver_types.h"
#if 0
typedef struct CUevent_st *
# 2758
cudaEvent_t; 
#endif
# 2763 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/driver_types.h"
#if 0
typedef cudaGraphicsResource *
# 2763
cudaGraphicsResource_t; 
#endif
# 2768 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/driver_types.h"
#if 0
typedef cudaOutputMode 
# 2768
cudaOutputMode_t; 
#endif
# 2773 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/driver_types.h"
#if 0
typedef struct CUexternalMemory_st *
# 2773
cudaExternalMemory_t; 
#endif
# 2778 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/driver_types.h"
#if 0
typedef struct CUexternalSemaphore_st *
# 2778
cudaExternalSemaphore_t; 
#endif
# 2783 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/driver_types.h"
#if 0
typedef struct CUgraph_st *
# 2783
cudaGraph_t; 
#endif
# 2788 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/driver_types.h"
#if 0
typedef struct CUgraphNode_st *
# 2788
cudaGraphNode_t; 
#endif
# 2793 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/driver_types.h"
#if 0
typedef struct CUuserObject_st *
# 2793
cudaUserObject_t; 
#endif
# 2798 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/driver_types.h"
#if 0
typedef struct CUfunc_st *
# 2798
cudaFunction_t; 
#endif
# 2803 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/driver_types.h"
#if 0
typedef struct CUmemPoolHandle_st *
# 2803
cudaMemPool_t; 
#endif
# 2808 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/driver_types.h"
#if 0
# 2808
enum cudaCGScope { 
# 2809
cudaCGScopeInvalid, 
# 2810
cudaCGScopeGrid, 
# 2811
cudaCGScopeMultiGrid
# 2812
}; 
#endif
# 2817 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/driver_types.h"
#if 0
# 2817
struct cudaLaunchParams { 
# 2819
void *func; 
# 2820
dim3 gridDim; 
# 2821
dim3 blockDim; 
# 2822
void **args; 
# 2823
size_t sharedMem; 
# 2824
cudaStream_t stream; 
# 2825
}; 
#endif
# 2830 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/driver_types.h"
#if 0
# 2830
struct cudaKernelNodeParams { 
# 2831
void *func; 
# 2832
dim3 gridDim; 
# 2833
dim3 blockDim; 
# 2834
unsigned sharedMemBytes; 
# 2835
void **kernelParams; 
# 2836
void **extra; 
# 2837
}; 
#endif
# 2842 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/driver_types.h"
#if 0
# 2842
struct cudaExternalSemaphoreSignalNodeParams { 
# 2843
cudaExternalSemaphore_t *extSemArray; 
# 2844
const cudaExternalSemaphoreSignalParams *paramsArray; 
# 2845
unsigned numExtSems; 
# 2846
}; 
#endif
# 2851 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/driver_types.h"
#if 0
# 2851
struct cudaExternalSemaphoreWaitNodeParams { 
# 2852
cudaExternalSemaphore_t *extSemArray; 
# 2853
const cudaExternalSemaphoreWaitParams *paramsArray; 
# 2854
unsigned numExtSems; 
# 2855
}; 
#endif
# 2860 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/driver_types.h"
#if 0
# 2860
enum cudaGraphNodeType { 
# 2861
cudaGraphNodeTypeKernel, 
# 2862
cudaGraphNodeTypeMemcpy, 
# 2863
cudaGraphNodeTypeMemset, 
# 2864
cudaGraphNodeTypeHost, 
# 2865
cudaGraphNodeTypeGraph, 
# 2866
cudaGraphNodeTypeEmpty, 
# 2867
cudaGraphNodeTypeWaitEvent, 
# 2868
cudaGraphNodeTypeEventRecord, 
# 2869
cudaGraphNodeTypeExtSemaphoreSignal, 
# 2870
cudaGraphNodeTypeExtSemaphoreWait, 
# 2871
cudaGraphNodeTypeMemAlloc, 
# 2872
cudaGraphNodeTypeMemFree, 
# 2873
cudaGraphNodeTypeCount
# 2874
}; 
#endif
# 2879 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/driver_types.h"
typedef struct CUgraphExec_st *cudaGraphExec_t; 
# 2884
#if 0
# 2884
enum cudaGraphExecUpdateResult { 
# 2885
cudaGraphExecUpdateSuccess, 
# 2886
cudaGraphExecUpdateError, 
# 2887
cudaGraphExecUpdateErrorTopologyChanged, 
# 2888
cudaGraphExecUpdateErrorNodeTypeChanged, 
# 2889
cudaGraphExecUpdateErrorFunctionChanged, 
# 2890
cudaGraphExecUpdateErrorParametersChanged, 
# 2891
cudaGraphExecUpdateErrorNotSupported, 
# 2892
cudaGraphExecUpdateErrorUnsupportedFunctionChange
# 2893
}; 
#endif
# 2899 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/driver_types.h"
#if 0
# 2899
enum cudaGetDriverEntryPointFlags { 
# 2900
cudaEnableDefault, 
# 2901
cudaEnableLegacyStream, 
# 2902
cudaEnablePerThreadDefaultStream
# 2903
}; 
#endif
# 2908 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/driver_types.h"
#if 0
# 2908
enum cudaGraphDebugDotFlags { 
# 2909
cudaGraphDebugDotFlagsVerbose = (1 << 0), 
# 2910
cudaGraphDebugDotFlagsKernelNodeParams = (1 << 2), 
# 2911
cudaGraphDebugDotFlagsMemcpyNodeParams = (1 << 3), 
# 2912
cudaGraphDebugDotFlagsMemsetNodeParams = (1 << 4), 
# 2913
cudaGraphDebugDotFlagsHostNodeParams = (1 << 5), 
# 2914
cudaGraphDebugDotFlagsEventNodeParams = (1 << 6), 
# 2915
cudaGraphDebugDotFlagsExtSemasSignalNodeParams = (1 << 7), 
# 2916
cudaGraphDebugDotFlagsExtSemasWaitNodeParams = (1 << 8), 
# 2917
cudaGraphDebugDotFlagsKernelNodeAttributes = (1 << 9), 
# 2918
cudaGraphDebugDotFlagsHandles = (1 << 10)
# 2919
}; 
#endif
# 2924 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/driver_types.h"
#if 0
# 2924
enum cudaGraphInstantiateFlags { 
# 2925
cudaGraphInstantiateFlagAutoFreeOnLaunch = 1
# 2926
}; 
#endif
# 84 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/surface_types.h"
#if 0
# 84
enum cudaSurfaceBoundaryMode { 
# 86
cudaBoundaryModeZero, 
# 87
cudaBoundaryModeClamp, 
# 88
cudaBoundaryModeTrap
# 89
}; 
#endif
# 94 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/surface_types.h"
#if 0
# 94
enum cudaSurfaceFormatMode { 
# 96
cudaFormatModeForced, 
# 97
cudaFormatModeAuto
# 98
}; 
#endif
# 103 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/surface_types.h"
#if 0
# 103
struct surfaceReference { 
# 108
cudaChannelFormatDesc channelDesc; 
# 109
}; 
#endif
# 114 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/surface_types.h"
#if 0
typedef unsigned long long 
# 114
cudaSurfaceObject_t; 
#endif
# 84 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/texture_types.h"
#if 0
# 84
enum cudaTextureAddressMode { 
# 86
cudaAddressModeWrap, 
# 87
cudaAddressModeClamp, 
# 88
cudaAddressModeMirror, 
# 89
cudaAddressModeBorder
# 90
}; 
#endif
# 95 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/texture_types.h"
#if 0
# 95
enum cudaTextureFilterMode { 
# 97
cudaFilterModePoint, 
# 98
cudaFilterModeLinear
# 99
}; 
#endif
# 104 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/texture_types.h"
#if 0
# 104
enum cudaTextureReadMode { 
# 106
cudaReadModeElementType, 
# 107
cudaReadModeNormalizedFloat
# 108
}; 
#endif
# 113 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/texture_types.h"
#if 0
# 113
struct textureReference { 
# 118
int normalized; 
# 122
cudaTextureFilterMode filterMode; 
# 126
cudaTextureAddressMode addressMode[3]; 
# 130
cudaChannelFormatDesc channelDesc; 
# 134
int sRGB; 
# 138
unsigned maxAnisotropy; 
# 142
cudaTextureFilterMode mipmapFilterMode; 
# 146
float mipmapLevelBias; 
# 150
float minMipmapLevelClamp; 
# 154
float maxMipmapLevelClamp; 
# 158
int disableTrilinearOptimization; 
# 159
int __cudaReserved[14]; 
# 160
}; 
#endif
# 165 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/texture_types.h"
#if 0
# 165
struct cudaTextureDesc { 
# 170
cudaTextureAddressMode addressMode[3]; 
# 174
cudaTextureFilterMode filterMode; 
# 178
cudaTextureReadMode readMode; 
# 182
int sRGB; 
# 186
float borderColor[4]; 
# 190
int normalizedCoords; 
# 194
unsigned maxAnisotropy; 
# 198
cudaTextureFilterMode mipmapFilterMode; 
# 202
float mipmapLevelBias; 
# 206
float minMipmapLevelClamp; 
# 210
float maxMipmapLevelClamp; 
# 214
int disableTrilinearOptimization; 
# 215
}; 
#endif
# 220 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/texture_types.h"
#if 0
typedef unsigned long long 
# 220
cudaTextureObject_t; 
#endif
# 84 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/library_types.h"
typedef 
# 54
enum cudaDataType_t { 
# 56
CUDA_R_16F = 2, 
# 57
CUDA_C_16F = 6, 
# 58
CUDA_R_16BF = 14, 
# 59
CUDA_C_16BF, 
# 60
CUDA_R_32F = 0, 
# 61
CUDA_C_32F = 4, 
# 62
CUDA_R_64F = 1, 
# 63
CUDA_C_64F = 5, 
# 64
CUDA_R_4I = 16, 
# 65
CUDA_C_4I, 
# 66
CUDA_R_4U, 
# 67
CUDA_C_4U, 
# 68
CUDA_R_8I = 3, 
# 69
CUDA_C_8I = 7, 
# 70
CUDA_R_8U, 
# 71
CUDA_C_8U, 
# 72
CUDA_R_16I = 20, 
# 73
CUDA_C_16I, 
# 74
CUDA_R_16U, 
# 75
CUDA_C_16U, 
# 76
CUDA_R_32I = 10, 
# 77
CUDA_C_32I, 
# 78
CUDA_R_32U, 
# 79
CUDA_C_32U, 
# 80
CUDA_R_64I = 24, 
# 81
CUDA_C_64I, 
# 82
CUDA_R_64U, 
# 83
CUDA_C_64U
# 84
} cudaDataType; 
# 92
typedef 
# 87
enum libraryPropertyType_t { 
# 89
MAJOR_VERSION, 
# 90
MINOR_VERSION, 
# 91
PATCH_LEVEL
# 92
} libraryPropertyType; 
# 115 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/cuda_device_runtime_api.h"
extern "C" {
# 117
extern cudaError_t cudaDeviceGetAttribute(int * value, cudaDeviceAttr attr, int device); 
# 118
extern cudaError_t cudaDeviceGetLimit(size_t * pValue, cudaLimit limit); 
# 119
extern cudaError_t cudaDeviceGetCacheConfig(cudaFuncCache * pCacheConfig); 
# 120
extern cudaError_t cudaDeviceGetSharedMemConfig(cudaSharedMemConfig * pConfig); 
# 121
extern cudaError_t cudaDeviceSynchronize(); 
# 122
extern cudaError_t cudaGetLastError(); 
# 123
extern cudaError_t cudaPeekAtLastError(); 
# 124
extern const char *cudaGetErrorString(cudaError_t error); 
# 125
extern const char *cudaGetErrorName(cudaError_t error); 
# 126
extern cudaError_t cudaGetDeviceCount(int * count); 
# 127
extern cudaError_t cudaGetDevice(int * device); 
# 128
extern cudaError_t cudaStreamCreateWithFlags(cudaStream_t * pStream, unsigned flags); 
# 129
extern cudaError_t cudaStreamDestroy(cudaStream_t stream); 
# 130
extern cudaError_t cudaStreamWaitEvent(cudaStream_t stream, cudaEvent_t event, unsigned flags); 
# 131
__attribute__((unused)) extern cudaError_t cudaStreamWaitEvent_ptsz(cudaStream_t stream, cudaEvent_t event, unsigned flags); 
# 132
extern cudaError_t cudaEventCreateWithFlags(cudaEvent_t * event, unsigned flags); 
# 133
extern cudaError_t cudaEventRecord(cudaEvent_t event, cudaStream_t stream); 
# 134
__attribute__((unused)) extern cudaError_t cudaEventRecord_ptsz(cudaEvent_t event, cudaStream_t stream); 
# 135
extern cudaError_t cudaEventRecordWithFlags(cudaEvent_t event, cudaStream_t stream, unsigned flags); 
# 136
__attribute__((unused)) extern cudaError_t cudaEventRecordWithFlags_ptsz(cudaEvent_t event, cudaStream_t stream, unsigned flags); 
# 137
extern cudaError_t cudaEventDestroy(cudaEvent_t event); 
# 138
extern cudaError_t cudaFuncGetAttributes(cudaFuncAttributes * attr, const void * func); 
# 139
extern cudaError_t cudaFree(void * devPtr); 
# 140
extern cudaError_t cudaMalloc(void ** devPtr, size_t size); 
# 141
extern cudaError_t cudaMemcpyAsync(void * dst, const void * src, size_t count, cudaMemcpyKind kind, cudaStream_t stream); 
# 142
__attribute__((unused)) extern cudaError_t cudaMemcpyAsync_ptsz(void * dst, const void * src, size_t count, cudaMemcpyKind kind, cudaStream_t stream); 
# 143
extern cudaError_t cudaMemcpy2DAsync(void * dst, size_t dpitch, const void * src, size_t spitch, size_t width, size_t height, cudaMemcpyKind kind, cudaStream_t stream); 
# 144
__attribute__((unused)) extern cudaError_t cudaMemcpy2DAsync_ptsz(void * dst, size_t dpitch, const void * src, size_t spitch, size_t width, size_t height, cudaMemcpyKind kind, cudaStream_t stream); 
# 145
extern cudaError_t cudaMemcpy3DAsync(const cudaMemcpy3DParms * p, cudaStream_t stream); 
# 146
__attribute__((unused)) extern cudaError_t cudaMemcpy3DAsync_ptsz(const cudaMemcpy3DParms * p, cudaStream_t stream); 
# 147
extern cudaError_t cudaMemsetAsync(void * devPtr, int value, size_t count, cudaStream_t stream); 
# 148
__attribute__((unused)) extern cudaError_t cudaMemsetAsync_ptsz(void * devPtr, int value, size_t count, cudaStream_t stream); 
# 149
extern cudaError_t cudaMemset2DAsync(void * devPtr, size_t pitch, int value, size_t width, size_t height, cudaStream_t stream); 
# 150
__attribute__((unused)) extern cudaError_t cudaMemset2DAsync_ptsz(void * devPtr, size_t pitch, int value, size_t width, size_t height, cudaStream_t stream); 
# 151
extern cudaError_t cudaMemset3DAsync(cudaPitchedPtr pitchedDevPtr, int value, cudaExtent extent, cudaStream_t stream); 
# 152
__attribute__((unused)) extern cudaError_t cudaMemset3DAsync_ptsz(cudaPitchedPtr pitchedDevPtr, int value, cudaExtent extent, cudaStream_t stream); 
# 153
extern cudaError_t cudaRuntimeGetVersion(int * runtimeVersion); 
# 174 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/cuda_device_runtime_api.h"
__attribute__((unused)) extern void *cudaGetParameterBuffer(size_t alignment, size_t size); 
# 202 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/cuda_device_runtime_api.h"
__attribute__((unused)) extern void *cudaGetParameterBufferV2(void * func, dim3 gridDimension, dim3 blockDimension, unsigned sharedMemSize); 
# 203
__attribute__((unused)) extern cudaError_t cudaLaunchDevice_ptsz(void * func, void * parameterBuffer, dim3 gridDimension, dim3 blockDimension, unsigned sharedMemSize, cudaStream_t stream); 
# 204
__attribute__((unused)) extern cudaError_t cudaLaunchDeviceV2_ptsz(void * parameterBuffer, cudaStream_t stream); 
# 222 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/cuda_device_runtime_api.h"
__attribute__((unused)) extern cudaError_t cudaLaunchDevice(void * func, void * parameterBuffer, dim3 gridDimension, dim3 blockDimension, unsigned sharedMemSize, cudaStream_t stream); 
# 223
__attribute__((unused)) extern cudaError_t cudaLaunchDeviceV2(void * parameterBuffer, cudaStream_t stream); 
# 226
extern cudaError_t cudaOccupancyMaxActiveBlocksPerMultiprocessor(int * numBlocks, const void * func, int blockSize, size_t dynamicSmemSize); 
# 227
extern cudaError_t cudaOccupancyMaxActiveBlocksPerMultiprocessorWithFlags(int * numBlocks, const void * func, int blockSize, size_t dynamicSmemSize, unsigned flags); 
# 229
__attribute__((unused)) extern unsigned long long cudaCGGetIntrinsicHandle(cudaCGScope scope); 
# 230
__attribute__((unused)) extern cudaError_t cudaCGSynchronize(unsigned long long handle, unsigned flags); 
# 231
__attribute__((unused)) extern cudaError_t cudaCGSynchronizeGrid(unsigned long long handle, unsigned flags); 
# 232
__attribute__((unused)) extern cudaError_t cudaCGGetSize(unsigned * numThreads, unsigned * numGrids, unsigned long long handle); 
# 233
__attribute__((unused)) extern cudaError_t cudaCGGetRank(unsigned * threadRank, unsigned * gridRank, unsigned long long handle); 
# 234
}
# 236
template< class T> static inline cudaError_t cudaMalloc(T ** devPtr, size_t size); 
# 237
template< class T> static inline cudaError_t cudaFuncGetAttributes(cudaFuncAttributes * attr, T * entry); 
# 238
template< class T> static inline cudaError_t cudaOccupancyMaxActiveBlocksPerMultiprocessor(int * numBlocks, T func, int blockSize, size_t dynamicSmemSize); 
# 239
template< class T> static inline cudaError_t cudaOccupancyMaxActiveBlocksPerMultiprocessorWithFlags(int * numBlocks, T func, int blockSize, size_t dynamicSmemSize, unsigned flags); 
# 266 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/cuda_runtime_api.h"
extern "C" {
# 301 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/cuda_runtime_api.h"
extern cudaError_t cudaDeviceReset(); 
# 322 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/cuda_runtime_api.h"
extern cudaError_t cudaDeviceSynchronize(); 
# 409 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/cuda_runtime_api.h"
extern cudaError_t cudaDeviceSetLimit(cudaLimit limit, size_t value); 
# 444 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/cuda_runtime_api.h"
extern cudaError_t cudaDeviceGetLimit(size_t * pValue, cudaLimit limit); 
# 467 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/cuda_runtime_api.h"
extern cudaError_t cudaDeviceGetTexture1DLinearMaxWidth(size_t * maxWidthInElements, const cudaChannelFormatDesc * fmtDesc, int device); 
# 501 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/cuda_runtime_api.h"
extern cudaError_t cudaDeviceGetCacheConfig(cudaFuncCache * pCacheConfig); 
# 538 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/cuda_runtime_api.h"
extern cudaError_t cudaDeviceGetStreamPriorityRange(int * leastPriority, int * greatestPriority); 
# 582 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/cuda_runtime_api.h"
extern cudaError_t cudaDeviceSetCacheConfig(cudaFuncCache cacheConfig); 
# 613 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/cuda_runtime_api.h"
extern cudaError_t cudaDeviceGetSharedMemConfig(cudaSharedMemConfig * pConfig); 
# 657 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/cuda_runtime_api.h"
extern cudaError_t cudaDeviceSetSharedMemConfig(cudaSharedMemConfig config); 
# 684 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/cuda_runtime_api.h"
extern cudaError_t cudaDeviceGetByPCIBusId(int * device, const char * pciBusId); 
# 714 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/cuda_runtime_api.h"
extern cudaError_t cudaDeviceGetPCIBusId(char * pciBusId, int len, int device); 
# 762 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/cuda_runtime_api.h"
extern cudaError_t cudaIpcGetEventHandle(cudaIpcEventHandle_t * handle, cudaEvent_t event); 
# 803 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/cuda_runtime_api.h"
extern cudaError_t cudaIpcOpenEventHandle(cudaEvent_t * event, cudaIpcEventHandle_t handle); 
# 846 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/cuda_runtime_api.h"
extern cudaError_t cudaIpcGetMemHandle(cudaIpcMemHandle_t * handle, void * devPtr); 
# 910 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/cuda_runtime_api.h"
extern cudaError_t cudaIpcOpenMemHandle(void ** devPtr, cudaIpcMemHandle_t handle, unsigned flags); 
# 946 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/cuda_runtime_api.h"
extern cudaError_t cudaIpcCloseMemHandle(void * devPtr); 
# 978 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/cuda_runtime_api.h"
extern cudaError_t cudaDeviceFlushGPUDirectRDMAWrites(cudaFlushGPUDirectRDMAWritesTarget target, cudaFlushGPUDirectRDMAWritesScope scope); 
# 1022 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/cuda_runtime_api.h"
__attribute((deprecated)) extern cudaError_t cudaThreadExit(); 
# 1048 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/cuda_runtime_api.h"
__attribute((deprecated)) extern cudaError_t cudaThreadSynchronize(); 
# 1097 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/cuda_runtime_api.h"
__attribute((deprecated)) extern cudaError_t cudaThreadSetLimit(cudaLimit limit, size_t value); 
# 1130 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/cuda_runtime_api.h"
__attribute((deprecated)) extern cudaError_t cudaThreadGetLimit(size_t * pValue, cudaLimit limit); 
# 1166 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/cuda_runtime_api.h"
__attribute((deprecated)) extern cudaError_t cudaThreadGetCacheConfig(cudaFuncCache * pCacheConfig); 
# 1213 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/cuda_runtime_api.h"
__attribute((deprecated)) extern cudaError_t cudaThreadSetCacheConfig(cudaFuncCache cacheConfig); 
# 1276 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/cuda_runtime_api.h"
extern cudaError_t cudaGetLastError(); 
# 1324 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/cuda_runtime_api.h"
extern cudaError_t cudaPeekAtLastError(); 
# 1340 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/cuda_runtime_api.h"
extern const char *cudaGetErrorName(cudaError_t error); 
# 1356 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/cuda_runtime_api.h"
extern const char *cudaGetErrorString(cudaError_t error); 
# 1384 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/cuda_runtime_api.h"
extern cudaError_t cudaGetDeviceCount(int * count); 
# 1662 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/cuda_runtime_api.h"
extern cudaError_t cudaGetDeviceProperties(cudaDeviceProp * prop, int device); 
# 1857 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/cuda_runtime_api.h"
extern cudaError_t cudaDeviceGetAttribute(int * value, cudaDeviceAttr attr, int device); 
# 1875 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/cuda_runtime_api.h"
extern cudaError_t cudaDeviceGetDefaultMemPool(cudaMemPool_t * memPool, int device); 
# 1899 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/cuda_runtime_api.h"
extern cudaError_t cudaDeviceSetMemPool(int device, cudaMemPool_t memPool); 
# 1919 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/cuda_runtime_api.h"
extern cudaError_t cudaDeviceGetMemPool(cudaMemPool_t * memPool, int device); 
# 1967 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/cuda_runtime_api.h"
extern cudaError_t cudaDeviceGetNvSciSyncAttributes(void * nvSciSyncAttrList, int device, int flags); 
# 2007 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/cuda_runtime_api.h"
extern cudaError_t cudaDeviceGetP2PAttribute(int * value, cudaDeviceP2PAttr attr, int srcDevice, int dstDevice); 
# 2028 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/cuda_runtime_api.h"
extern cudaError_t cudaChooseDevice(int * device, const cudaDeviceProp * prop); 
# 2065 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/cuda_runtime_api.h"
extern cudaError_t cudaSetDevice(int device); 
# 2086 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/cuda_runtime_api.h"
extern cudaError_t cudaGetDevice(int * device); 
# 2117 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/cuda_runtime_api.h"
extern cudaError_t cudaSetValidDevices(int * device_arr, int len); 
# 2182 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/cuda_runtime_api.h"
extern cudaError_t cudaSetDeviceFlags(unsigned flags); 
# 2226 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/cuda_runtime_api.h"
extern cudaError_t cudaGetDeviceFlags(unsigned * flags); 
# 2266 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/cuda_runtime_api.h"
extern cudaError_t cudaStreamCreate(cudaStream_t * pStream); 
# 2298 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/cuda_runtime_api.h"
extern cudaError_t cudaStreamCreateWithFlags(cudaStream_t * pStream, unsigned flags); 
# 2344 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/cuda_runtime_api.h"
extern cudaError_t cudaStreamCreateWithPriority(cudaStream_t * pStream, unsigned flags, int priority); 
# 2371 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/cuda_runtime_api.h"
extern cudaError_t cudaStreamGetPriority(cudaStream_t hStream, int * priority); 
# 2396 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/cuda_runtime_api.h"
extern cudaError_t cudaStreamGetFlags(cudaStream_t hStream, unsigned * flags); 
# 2411 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/cuda_runtime_api.h"
extern cudaError_t cudaCtxResetPersistingL2Cache(); 
# 2431 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/cuda_runtime_api.h"
extern cudaError_t cudaStreamCopyAttributes(cudaStream_t dst, cudaStream_t src); 
# 2452 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/cuda_runtime_api.h"
extern cudaError_t cudaStreamGetAttribute(cudaStream_t hStream, cudaStreamAttrID attr, cudaStreamAttrValue * value_out); 
# 2476 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/cuda_runtime_api.h"
extern cudaError_t cudaStreamSetAttribute(cudaStream_t hStream, cudaStreamAttrID attr, const cudaStreamAttrValue * value); 
# 2510 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/cuda_runtime_api.h"
extern cudaError_t cudaStreamDestroy(cudaStream_t stream); 
# 2541 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/cuda_runtime_api.h"
extern cudaError_t cudaStreamWaitEvent(cudaStream_t stream, cudaEvent_t event, unsigned flags = 0); 
# 2549
typedef void (*cudaStreamCallback_t)(cudaStream_t stream, cudaError_t status, void * userData); 
# 2616 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/cuda_runtime_api.h"
extern cudaError_t cudaStreamAddCallback(cudaStream_t stream, cudaStreamCallback_t callback, void * userData, unsigned flags); 
# 2640 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/cuda_runtime_api.h"
extern cudaError_t cudaStreamSynchronize(cudaStream_t stream); 
# 2665 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/cuda_runtime_api.h"
extern cudaError_t cudaStreamQuery(cudaStream_t stream); 
# 2749 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/cuda_runtime_api.h"
extern cudaError_t cudaStreamAttachMemAsync(cudaStream_t stream, void * devPtr, size_t length = 0, unsigned flags = 4); 
# 2788 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/cuda_runtime_api.h"
extern cudaError_t cudaStreamBeginCapture(cudaStream_t stream, cudaStreamCaptureMode mode); 
# 2839 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/cuda_runtime_api.h"
extern cudaError_t cudaThreadExchangeStreamCaptureMode(cudaStreamCaptureMode * mode); 
# 2867 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/cuda_runtime_api.h"
extern cudaError_t cudaStreamEndCapture(cudaStream_t stream, cudaGraph_t * pGraph); 
# 2905 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/cuda_runtime_api.h"
extern cudaError_t cudaStreamIsCapturing(cudaStream_t stream, cudaStreamCaptureStatus * pCaptureStatus); 
# 2937 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/cuda_runtime_api.h"
extern cudaError_t cudaStreamGetCaptureInfo(cudaStream_t stream, cudaStreamCaptureStatus * pCaptureStatus, unsigned long long * pId); 
# 2992 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/cuda_runtime_api.h"
extern cudaError_t cudaStreamGetCaptureInfo_v2(cudaStream_t stream, cudaStreamCaptureStatus * captureStatus_out, unsigned long long * id_out = 0, cudaGraph_t * graph_out = 0, const cudaGraphNode_t ** dependencies_out = 0, size_t * numDependencies_out = 0); 
# 3025 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/cuda_runtime_api.h"
extern cudaError_t cudaStreamUpdateCaptureDependencies(cudaStream_t stream, cudaGraphNode_t * dependencies, size_t numDependencies, unsigned flags = 0); 
# 3062 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/cuda_runtime_api.h"
extern cudaError_t cudaEventCreate(cudaEvent_t * event); 
# 3099 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/cuda_runtime_api.h"
extern cudaError_t cudaEventCreateWithFlags(cudaEvent_t * event, unsigned flags); 
# 3139 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/cuda_runtime_api.h"
extern cudaError_t cudaEventRecord(cudaEvent_t event, cudaStream_t stream = 0); 
# 3186 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/cuda_runtime_api.h"
extern cudaError_t cudaEventRecordWithFlags(cudaEvent_t event, cudaStream_t stream = 0, unsigned flags = 0); 
# 3218 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/cuda_runtime_api.h"
extern cudaError_t cudaEventQuery(cudaEvent_t event); 
# 3248 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/cuda_runtime_api.h"
extern cudaError_t cudaEventSynchronize(cudaEvent_t event); 
# 3277 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/cuda_runtime_api.h"
extern cudaError_t cudaEventDestroy(cudaEvent_t event); 
# 3320 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/cuda_runtime_api.h"
extern cudaError_t cudaEventElapsedTime(float * ms, cudaEvent_t start, cudaEvent_t end); 
# 3497 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/cuda_runtime_api.h"
extern cudaError_t cudaImportExternalMemory(cudaExternalMemory_t * extMem_out, const cudaExternalMemoryHandleDesc * memHandleDesc); 
# 3551 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/cuda_runtime_api.h"
extern cudaError_t cudaExternalMemoryGetMappedBuffer(void ** devPtr, cudaExternalMemory_t extMem, const cudaExternalMemoryBufferDesc * bufferDesc); 
# 3610 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/cuda_runtime_api.h"
extern cudaError_t cudaExternalMemoryGetMappedMipmappedArray(cudaMipmappedArray_t * mipmap, cudaExternalMemory_t extMem, const cudaExternalMemoryMipmappedArrayDesc * mipmapDesc); 
# 3634 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/cuda_runtime_api.h"
extern cudaError_t cudaDestroyExternalMemory(cudaExternalMemory_t extMem); 
# 3787 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/cuda_runtime_api.h"
extern cudaError_t cudaImportExternalSemaphore(cudaExternalSemaphore_t * extSem_out, const cudaExternalSemaphoreHandleDesc * semHandleDesc); 
# 3854 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/cuda_runtime_api.h"
extern cudaError_t cudaSignalExternalSemaphoresAsync_v2(const cudaExternalSemaphore_t * extSemArray, const cudaExternalSemaphoreSignalParams * paramsArray, unsigned numExtSems, cudaStream_t stream = 0); 
# 3930 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/cuda_runtime_api.h"
extern cudaError_t cudaWaitExternalSemaphoresAsync_v2(const cudaExternalSemaphore_t * extSemArray, const cudaExternalSemaphoreWaitParams * paramsArray, unsigned numExtSems, cudaStream_t stream = 0); 
# 3953 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/cuda_runtime_api.h"
extern cudaError_t cudaDestroyExternalSemaphore(cudaExternalSemaphore_t extSem); 
# 4020 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/cuda_runtime_api.h"
extern cudaError_t cudaLaunchKernel(const void * func, dim3 gridDim, dim3 blockDim, void ** args, size_t sharedMem, cudaStream_t stream); 
# 4077 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/cuda_runtime_api.h"
extern cudaError_t cudaLaunchCooperativeKernel(const void * func, dim3 gridDim, dim3 blockDim, void ** args, size_t sharedMem, cudaStream_t stream); 
# 4178 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/cuda_runtime_api.h"
__attribute((deprecated)) extern cudaError_t cudaLaunchCooperativeKernelMultiDevice(cudaLaunchParams * launchParamsList, unsigned numDevices, unsigned flags = 0); 
# 4225 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/cuda_runtime_api.h"
extern cudaError_t cudaFuncSetCacheConfig(const void * func, cudaFuncCache cacheConfig); 
# 4280 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/cuda_runtime_api.h"
extern cudaError_t cudaFuncSetSharedMemConfig(const void * func, cudaSharedMemConfig config); 
# 4313 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/cuda_runtime_api.h"
extern cudaError_t cudaFuncGetAttributes(cudaFuncAttributes * attr, const void * func); 
# 4350 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/cuda_runtime_api.h"
extern cudaError_t cudaFuncSetAttribute(const void * func, cudaFuncAttribute attr, int value); 
# 4376 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/cuda_runtime_api.h"
__attribute((deprecated)) extern cudaError_t cudaSetDoubleForDevice(double * d); 
# 4400 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/cuda_runtime_api.h"
__attribute((deprecated)) extern cudaError_t cudaSetDoubleForHost(double * d); 
# 4468 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/cuda_runtime_api.h"
extern cudaError_t cudaLaunchHostFunc(cudaStream_t stream, cudaHostFn_t fn, void * userData); 
# 4525 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/cuda_runtime_api.h"
extern cudaError_t cudaOccupancyMaxActiveBlocksPerMultiprocessor(int * numBlocks, const void * func, int blockSize, size_t dynamicSMemSize); 
# 4554 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/cuda_runtime_api.h"
extern cudaError_t cudaOccupancyAvailableDynamicSMemPerBlock(size_t * dynamicSmemSize, const void * func, int numBlocks, int blockSize); 
# 4599 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/cuda_runtime_api.h"
extern cudaError_t cudaOccupancyMaxActiveBlocksPerMultiprocessorWithFlags(int * numBlocks, const void * func, int blockSize, size_t dynamicSMemSize, unsigned flags); 
# 4720 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/cuda_runtime_api.h"
extern cudaError_t cudaMallocManaged(void ** devPtr, size_t size, unsigned flags = 1); 
# 4753 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/cuda_runtime_api.h"
extern cudaError_t cudaMalloc(void ** devPtr, size_t size); 
# 4786 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/cuda_runtime_api.h"
extern cudaError_t cudaMallocHost(void ** ptr, size_t size); 
# 4829 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/cuda_runtime_api.h"
extern cudaError_t cudaMallocPitch(void ** devPtr, size_t * pitch, size_t width, size_t height); 
# 4878 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/cuda_runtime_api.h"
extern cudaError_t cudaMallocArray(cudaArray_t * array, const cudaChannelFormatDesc * desc, size_t width, size_t height = 0, unsigned flags = 0); 
# 4907 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/cuda_runtime_api.h"
extern cudaError_t cudaFree(void * devPtr); 
# 4930 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/cuda_runtime_api.h"
extern cudaError_t cudaFreeHost(void * ptr); 
# 4953 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/cuda_runtime_api.h"
extern cudaError_t cudaFreeArray(cudaArray_t array); 
# 4976 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/cuda_runtime_api.h"
extern cudaError_t cudaFreeMipmappedArray(cudaMipmappedArray_t mipmappedArray); 
# 5042 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/cuda_runtime_api.h"
extern cudaError_t cudaHostAlloc(void ** pHost, size_t size, unsigned flags); 
# 5135 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/cuda_runtime_api.h"
extern cudaError_t cudaHostRegister(void * ptr, size_t size, unsigned flags); 
# 5158 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/cuda_runtime_api.h"
extern cudaError_t cudaHostUnregister(void * ptr); 
# 5203 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/cuda_runtime_api.h"
extern cudaError_t cudaHostGetDevicePointer(void ** pDevice, void * pHost, unsigned flags); 
# 5225 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/cuda_runtime_api.h"
extern cudaError_t cudaHostGetFlags(unsigned * pFlags, void * pHost); 
# 5264 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/cuda_runtime_api.h"
extern cudaError_t cudaMalloc3D(cudaPitchedPtr * pitchedDevPtr, cudaExtent extent); 
# 5406 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/cuda_runtime_api.h"
extern cudaError_t cudaMalloc3DArray(cudaArray_t * array, const cudaChannelFormatDesc * desc, cudaExtent extent, unsigned flags = 0); 
# 5548 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/cuda_runtime_api.h"
extern cudaError_t cudaMallocMipmappedArray(cudaMipmappedArray_t * mipmappedArray, const cudaChannelFormatDesc * desc, cudaExtent extent, unsigned numLevels, unsigned flags = 0); 
# 5581 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/cuda_runtime_api.h"
extern cudaError_t cudaGetMipmappedArrayLevel(cudaArray_t * levelArray, cudaMipmappedArray_const_t mipmappedArray, unsigned level); 
# 5686 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/cuda_runtime_api.h"
extern cudaError_t cudaMemcpy3D(const cudaMemcpy3DParms * p); 
# 5717 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/cuda_runtime_api.h"
extern cudaError_t cudaMemcpy3DPeer(const cudaMemcpy3DPeerParms * p); 
# 5835 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/cuda_runtime_api.h"
extern cudaError_t cudaMemcpy3DAsync(const cudaMemcpy3DParms * p, cudaStream_t stream = 0); 
# 5861 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/cuda_runtime_api.h"
extern cudaError_t cudaMemcpy3DPeerAsync(const cudaMemcpy3DPeerParms * p, cudaStream_t stream = 0); 
# 5883 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/cuda_runtime_api.h"
extern cudaError_t cudaMemGetInfo(size_t * free, size_t * total); 
# 5909 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/cuda_runtime_api.h"
extern cudaError_t cudaArrayGetInfo(cudaChannelFormatDesc * desc, cudaExtent * extent, unsigned * flags, cudaArray_t array); 
# 5938 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/cuda_runtime_api.h"
extern cudaError_t cudaArrayGetPlane(cudaArray_t * pPlaneArray, cudaArray_t hArray, unsigned planeIdx); 
# 5966 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/cuda_runtime_api.h"
extern cudaError_t cudaArrayGetSparseProperties(cudaArraySparseProperties * sparseProperties, cudaArray_t array); 
# 5996 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/cuda_runtime_api.h"
extern cudaError_t cudaMipmappedArrayGetSparseProperties(cudaArraySparseProperties * sparseProperties, cudaMipmappedArray_t mipmap); 
# 6041 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/cuda_runtime_api.h"
extern cudaError_t cudaMemcpy(void * dst, const void * src, size_t count, cudaMemcpyKind kind); 
# 6076 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/cuda_runtime_api.h"
extern cudaError_t cudaMemcpyPeer(void * dst, int dstDevice, const void * src, int srcDevice, size_t count); 
# 6125 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/cuda_runtime_api.h"
extern cudaError_t cudaMemcpy2D(void * dst, size_t dpitch, const void * src, size_t spitch, size_t width, size_t height, cudaMemcpyKind kind); 
# 6175 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/cuda_runtime_api.h"
extern cudaError_t cudaMemcpy2DToArray(cudaArray_t dst, size_t wOffset, size_t hOffset, const void * src, size_t spitch, size_t width, size_t height, cudaMemcpyKind kind); 
# 6225 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/cuda_runtime_api.h"
extern cudaError_t cudaMemcpy2DFromArray(void * dst, size_t dpitch, cudaArray_const_t src, size_t wOffset, size_t hOffset, size_t width, size_t height, cudaMemcpyKind kind); 
# 6272 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/cuda_runtime_api.h"
extern cudaError_t cudaMemcpy2DArrayToArray(cudaArray_t dst, size_t wOffsetDst, size_t hOffsetDst, cudaArray_const_t src, size_t wOffsetSrc, size_t hOffsetSrc, size_t width, size_t height, cudaMemcpyKind kind = cudaMemcpyDeviceToDevice); 
# 6315 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/cuda_runtime_api.h"
extern cudaError_t cudaMemcpyToSymbol(const void * symbol, const void * src, size_t count, size_t offset = 0, cudaMemcpyKind kind = cudaMemcpyHostToDevice); 
# 6358 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/cuda_runtime_api.h"
extern cudaError_t cudaMemcpyFromSymbol(void * dst, const void * symbol, size_t count, size_t offset = 0, cudaMemcpyKind kind = cudaMemcpyDeviceToHost); 
# 6415 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/cuda_runtime_api.h"
extern cudaError_t cudaMemcpyAsync(void * dst, const void * src, size_t count, cudaMemcpyKind kind, cudaStream_t stream = 0); 
# 6450 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/cuda_runtime_api.h"
extern cudaError_t cudaMemcpyPeerAsync(void * dst, int dstDevice, const void * src, int srcDevice, size_t count, cudaStream_t stream = 0); 
# 6513 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/cuda_runtime_api.h"
extern cudaError_t cudaMemcpy2DAsync(void * dst, size_t dpitch, const void * src, size_t spitch, size_t width, size_t height, cudaMemcpyKind kind, cudaStream_t stream = 0); 
# 6571 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/cuda_runtime_api.h"
extern cudaError_t cudaMemcpy2DToArrayAsync(cudaArray_t dst, size_t wOffset, size_t hOffset, const void * src, size_t spitch, size_t width, size_t height, cudaMemcpyKind kind, cudaStream_t stream = 0); 
# 6628 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/cuda_runtime_api.h"
extern cudaError_t cudaMemcpy2DFromArrayAsync(void * dst, size_t dpitch, cudaArray_const_t src, size_t wOffset, size_t hOffset, size_t width, size_t height, cudaMemcpyKind kind, cudaStream_t stream = 0); 
# 6679 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/cuda_runtime_api.h"
extern cudaError_t cudaMemcpyToSymbolAsync(const void * symbol, const void * src, size_t count, size_t offset, cudaMemcpyKind kind, cudaStream_t stream = 0); 
# 6730 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/cuda_runtime_api.h"
extern cudaError_t cudaMemcpyFromSymbolAsync(void * dst, const void * symbol, size_t count, size_t offset, cudaMemcpyKind kind, cudaStream_t stream = 0); 
# 6759 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/cuda_runtime_api.h"
extern cudaError_t cudaMemset(void * devPtr, int value, size_t count); 
# 6793 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/cuda_runtime_api.h"
extern cudaError_t cudaMemset2D(void * devPtr, size_t pitch, int value, size_t width, size_t height); 
# 6839 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/cuda_runtime_api.h"
extern cudaError_t cudaMemset3D(cudaPitchedPtr pitchedDevPtr, int value, cudaExtent extent); 
# 6875 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/cuda_runtime_api.h"
extern cudaError_t cudaMemsetAsync(void * devPtr, int value, size_t count, cudaStream_t stream = 0); 
# 6916 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/cuda_runtime_api.h"
extern cudaError_t cudaMemset2DAsync(void * devPtr, size_t pitch, int value, size_t width, size_t height, cudaStream_t stream = 0); 
# 6969 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/cuda_runtime_api.h"
extern cudaError_t cudaMemset3DAsync(cudaPitchedPtr pitchedDevPtr, int value, cudaExtent extent, cudaStream_t stream = 0); 
# 6997 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/cuda_runtime_api.h"
extern cudaError_t cudaGetSymbolAddress(void ** devPtr, const void * symbol); 
# 7024 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/cuda_runtime_api.h"
extern cudaError_t cudaGetSymbolSize(size_t * size, const void * symbol); 
# 7094 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/cuda_runtime_api.h"
extern cudaError_t cudaMemPrefetchAsync(const void * devPtr, size_t count, int dstDevice, cudaStream_t stream = 0); 
# 7210 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/cuda_runtime_api.h"
extern cudaError_t cudaMemAdvise(const void * devPtr, size_t count, cudaMemoryAdvise advice, int device); 
# 7269 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/cuda_runtime_api.h"
extern cudaError_t cudaMemRangeGetAttribute(void * data, size_t dataSize, cudaMemRangeAttribute attribute, const void * devPtr, size_t count); 
# 7308 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/cuda_runtime_api.h"
extern cudaError_t cudaMemRangeGetAttributes(void ** data, size_t * dataSizes, cudaMemRangeAttribute * attributes, size_t numAttributes, const void * devPtr, size_t count); 
# 7368 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/cuda_runtime_api.h"
__attribute((deprecated)) extern cudaError_t cudaMemcpyToArray(cudaArray_t dst, size_t wOffset, size_t hOffset, const void * src, size_t count, cudaMemcpyKind kind); 
# 7410 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/cuda_runtime_api.h"
__attribute((deprecated)) extern cudaError_t cudaMemcpyFromArray(void * dst, cudaArray_const_t src, size_t wOffset, size_t hOffset, size_t count, cudaMemcpyKind kind); 
# 7453 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/cuda_runtime_api.h"
__attribute((deprecated)) extern cudaError_t cudaMemcpyArrayToArray(cudaArray_t dst, size_t wOffsetDst, size_t hOffsetDst, cudaArray_const_t src, size_t wOffsetSrc, size_t hOffsetSrc, size_t count, cudaMemcpyKind kind = cudaMemcpyDeviceToDevice); 
# 7504 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/cuda_runtime_api.h"
__attribute((deprecated)) extern cudaError_t cudaMemcpyToArrayAsync(cudaArray_t dst, size_t wOffset, size_t hOffset, const void * src, size_t count, cudaMemcpyKind kind, cudaStream_t stream = 0); 
# 7554 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/cuda_runtime_api.h"
__attribute((deprecated)) extern cudaError_t cudaMemcpyFromArrayAsync(void * dst, cudaArray_const_t src, size_t wOffset, size_t hOffset, size_t count, cudaMemcpyKind kind, cudaStream_t stream = 0); 
# 7623 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/cuda_runtime_api.h"
extern cudaError_t cudaMallocAsync(void ** devPtr, size_t size, cudaStream_t hStream); 
# 7649 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/cuda_runtime_api.h"
extern cudaError_t cudaFreeAsync(void * devPtr, cudaStream_t hStream); 
# 7674 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/cuda_runtime_api.h"
extern cudaError_t cudaMemPoolTrimTo(cudaMemPool_t memPool, size_t minBytesToKeep); 
# 7712 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/cuda_runtime_api.h"
extern cudaError_t cudaMemPoolSetAttribute(cudaMemPool_t memPool, cudaMemPoolAttr attr, void * value); 
# 7750 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/cuda_runtime_api.h"
extern cudaError_t cudaMemPoolGetAttribute(cudaMemPool_t memPool, cudaMemPoolAttr attr, void * value); 
# 7765 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/cuda_runtime_api.h"
extern cudaError_t cudaMemPoolSetAccess(cudaMemPool_t memPool, const cudaMemAccessDesc * descList, size_t count); 
# 7778 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/cuda_runtime_api.h"
extern cudaError_t cudaMemPoolGetAccess(cudaMemAccessFlags * flags, cudaMemPool_t memPool, cudaMemLocation * location); 
# 7798 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/cuda_runtime_api.h"
extern cudaError_t cudaMemPoolCreate(cudaMemPool_t * memPool, const cudaMemPoolProps * poolProps); 
# 7820 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/cuda_runtime_api.h"
extern cudaError_t cudaMemPoolDestroy(cudaMemPool_t memPool); 
# 7856 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/cuda_runtime_api.h"
extern cudaError_t cudaMallocFromPoolAsync(void ** ptr, size_t size, cudaMemPool_t memPool, cudaStream_t stream); 
# 7881 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/cuda_runtime_api.h"
extern cudaError_t cudaMemPoolExportToShareableHandle(void * shareableHandle, cudaMemPool_t memPool, cudaMemAllocationHandleType handleType, unsigned flags); 
# 7908 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/cuda_runtime_api.h"
extern cudaError_t cudaMemPoolImportFromShareableHandle(cudaMemPool_t * memPool, void * shareableHandle, cudaMemAllocationHandleType handleType, unsigned flags); 
# 7931 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/cuda_runtime_api.h"
extern cudaError_t cudaMemPoolExportPointer(cudaMemPoolPtrExportData * exportData, void * ptr); 
# 7960 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/cuda_runtime_api.h"
extern cudaError_t cudaMemPoolImportPointer(void ** ptr, cudaMemPool_t memPool, cudaMemPoolPtrExportData * exportData); 
# 8112 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/cuda_runtime_api.h"
extern cudaError_t cudaPointerGetAttributes(cudaPointerAttributes * attributes, const void * ptr); 
# 8153 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/cuda_runtime_api.h"
extern cudaError_t cudaDeviceCanAccessPeer(int * canAccessPeer, int device, int peerDevice); 
# 8195 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/cuda_runtime_api.h"
extern cudaError_t cudaDeviceEnablePeerAccess(int peerDevice, unsigned flags); 
# 8217 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/cuda_runtime_api.h"
extern cudaError_t cudaDeviceDisablePeerAccess(int peerDevice); 
# 8281 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/cuda_runtime_api.h"
extern cudaError_t cudaGraphicsUnregisterResource(cudaGraphicsResource_t resource); 
# 8316 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/cuda_runtime_api.h"
extern cudaError_t cudaGraphicsResourceSetMapFlags(cudaGraphicsResource_t resource, unsigned flags); 
# 8355 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/cuda_runtime_api.h"
extern cudaError_t cudaGraphicsMapResources(int count, cudaGraphicsResource_t * resources, cudaStream_t stream = 0); 
# 8390 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/cuda_runtime_api.h"
extern cudaError_t cudaGraphicsUnmapResources(int count, cudaGraphicsResource_t * resources, cudaStream_t stream = 0); 
# 8422 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/cuda_runtime_api.h"
extern cudaError_t cudaGraphicsResourceGetMappedPointer(void ** devPtr, size_t * size, cudaGraphicsResource_t resource); 
# 8460 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/cuda_runtime_api.h"
extern cudaError_t cudaGraphicsSubResourceGetMappedArray(cudaArray_t * array, cudaGraphicsResource_t resource, unsigned arrayIndex, unsigned mipLevel); 
# 8489 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/cuda_runtime_api.h"
extern cudaError_t cudaGraphicsResourceGetMappedMipmappedArray(cudaMipmappedArray_t * mipmappedArray, cudaGraphicsResource_t resource); 
# 8560 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/cuda_runtime_api.h"
__attribute((deprecated)) extern cudaError_t cudaBindTexture(size_t * offset, const textureReference * texref, const void * devPtr, const cudaChannelFormatDesc * desc, size_t size = ((2147483647) * 2U) + 1U); 
# 8619 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/cuda_runtime_api.h"
__attribute((deprecated)) extern cudaError_t cudaBindTexture2D(size_t * offset, const textureReference * texref, const void * devPtr, const cudaChannelFormatDesc * desc, size_t width, size_t height, size_t pitch); 
# 8657 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/cuda_runtime_api.h"
__attribute((deprecated)) extern cudaError_t cudaBindTextureToArray(const textureReference * texref, cudaArray_const_t array, const cudaChannelFormatDesc * desc); 
# 8697 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/cuda_runtime_api.h"
__attribute((deprecated)) extern cudaError_t cudaBindTextureToMipmappedArray(const textureReference * texref, cudaMipmappedArray_const_t mipmappedArray, const cudaChannelFormatDesc * desc); 
# 8723 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/cuda_runtime_api.h"
__attribute((deprecated)) extern cudaError_t cudaUnbindTexture(const textureReference * texref); 
# 8752 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/cuda_runtime_api.h"
__attribute((deprecated)) extern cudaError_t cudaGetTextureAlignmentOffset(size_t * offset, const textureReference * texref); 
# 8782 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/cuda_runtime_api.h"
__attribute((deprecated)) extern cudaError_t cudaGetTextureReference(const textureReference ** texref, const void * symbol); 
# 8827 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/cuda_runtime_api.h"
__attribute((deprecated)) extern cudaError_t cudaBindSurfaceToArray(const surfaceReference * surfref, cudaArray_const_t array, const cudaChannelFormatDesc * desc); 
# 8852 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/cuda_runtime_api.h"
__attribute((deprecated)) extern cudaError_t cudaGetSurfaceReference(const surfaceReference ** surfref, const void * symbol); 
# 8887 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/cuda_runtime_api.h"
extern cudaError_t cudaGetChannelDesc(cudaChannelFormatDesc * desc, cudaArray_const_t array); 
# 8917 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/cuda_runtime_api.h"
extern cudaChannelFormatDesc cudaCreateChannelDesc(int x, int y, int z, int w, cudaChannelFormatKind f); 
# 9135 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/cuda_runtime_api.h"
extern cudaError_t cudaCreateTextureObject(cudaTextureObject_t * pTexObject, const cudaResourceDesc * pResDesc, const cudaTextureDesc * pTexDesc, const cudaResourceViewDesc * pResViewDesc); 
# 9155 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/cuda_runtime_api.h"
extern cudaError_t cudaDestroyTextureObject(cudaTextureObject_t texObject); 
# 9175 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/cuda_runtime_api.h"
extern cudaError_t cudaGetTextureObjectResourceDesc(cudaResourceDesc * pResDesc, cudaTextureObject_t texObject); 
# 9195 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/cuda_runtime_api.h"
extern cudaError_t cudaGetTextureObjectTextureDesc(cudaTextureDesc * pTexDesc, cudaTextureObject_t texObject); 
# 9216 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/cuda_runtime_api.h"
extern cudaError_t cudaGetTextureObjectResourceViewDesc(cudaResourceViewDesc * pResViewDesc, cudaTextureObject_t texObject); 
# 9261 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/cuda_runtime_api.h"
extern cudaError_t cudaCreateSurfaceObject(cudaSurfaceObject_t * pSurfObject, const cudaResourceDesc * pResDesc); 
# 9281 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/cuda_runtime_api.h"
extern cudaError_t cudaDestroySurfaceObject(cudaSurfaceObject_t surfObject); 
# 9300 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/cuda_runtime_api.h"
extern cudaError_t cudaGetSurfaceObjectResourceDesc(cudaResourceDesc * pResDesc, cudaSurfaceObject_t surfObject); 
# 9334 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/cuda_runtime_api.h"
extern cudaError_t cudaDriverGetVersion(int * driverVersion); 
# 9359 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/cuda_runtime_api.h"
extern cudaError_t cudaRuntimeGetVersion(int * runtimeVersion); 
# 9406 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/cuda_runtime_api.h"
extern cudaError_t cudaGraphCreate(cudaGraph_t * pGraph, unsigned flags); 
# 9503 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/cuda_runtime_api.h"
extern cudaError_t cudaGraphAddKernelNode(cudaGraphNode_t * pGraphNode, cudaGraph_t graph, const cudaGraphNode_t * pDependencies, size_t numDependencies, const cudaKernelNodeParams * pNodeParams); 
# 9536 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/cuda_runtime_api.h"
extern cudaError_t cudaGraphKernelNodeGetParams(cudaGraphNode_t node, cudaKernelNodeParams * pNodeParams); 
# 9561 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/cuda_runtime_api.h"
extern cudaError_t cudaGraphKernelNodeSetParams(cudaGraphNode_t node, const cudaKernelNodeParams * pNodeParams); 
# 9581 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/cuda_runtime_api.h"
extern cudaError_t cudaGraphKernelNodeCopyAttributes(cudaGraphNode_t hSrc, cudaGraphNode_t hDst); 
# 9604 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/cuda_runtime_api.h"
extern cudaError_t cudaGraphKernelNodeGetAttribute(cudaGraphNode_t hNode, cudaKernelNodeAttrID attr, cudaKernelNodeAttrValue * value_out); 
# 9628 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/cuda_runtime_api.h"
extern cudaError_t cudaGraphKernelNodeSetAttribute(cudaGraphNode_t hNode, cudaKernelNodeAttrID attr, const cudaKernelNodeAttrValue * value); 
# 9678 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/cuda_runtime_api.h"
extern cudaError_t cudaGraphAddMemcpyNode(cudaGraphNode_t * pGraphNode, cudaGraph_t graph, const cudaGraphNode_t * pDependencies, size_t numDependencies, const cudaMemcpy3DParms * pCopyParams); 
# 9737 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/cuda_runtime_api.h"
extern cudaError_t cudaGraphAddMemcpyNodeToSymbol(cudaGraphNode_t * pGraphNode, cudaGraph_t graph, const cudaGraphNode_t * pDependencies, size_t numDependencies, const void * symbol, const void * src, size_t count, size_t offset, cudaMemcpyKind kind); 
# 9806 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/cuda_runtime_api.h"
extern cudaError_t cudaGraphAddMemcpyNodeFromSymbol(cudaGraphNode_t * pGraphNode, cudaGraph_t graph, const cudaGraphNode_t * pDependencies, size_t numDependencies, void * dst, const void * symbol, size_t count, size_t offset, cudaMemcpyKind kind); 
# 9874 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/cuda_runtime_api.h"
extern cudaError_t cudaGraphAddMemcpyNode1D(cudaGraphNode_t * pGraphNode, cudaGraph_t graph, const cudaGraphNode_t * pDependencies, size_t numDependencies, void * dst, const void * src, size_t count, cudaMemcpyKind kind); 
# 9906 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/cuda_runtime_api.h"
extern cudaError_t cudaGraphMemcpyNodeGetParams(cudaGraphNode_t node, cudaMemcpy3DParms * pNodeParams); 
# 9932 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/cuda_runtime_api.h"
extern cudaError_t cudaGraphMemcpyNodeSetParams(cudaGraphNode_t node, const cudaMemcpy3DParms * pNodeParams); 
# 9971 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/cuda_runtime_api.h"
extern cudaError_t cudaGraphMemcpyNodeSetParamsToSymbol(cudaGraphNode_t node, const void * symbol, const void * src, size_t count, size_t offset, cudaMemcpyKind kind); 
# 10017 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/cuda_runtime_api.h"
extern cudaError_t cudaGraphMemcpyNodeSetParamsFromSymbol(cudaGraphNode_t node, void * dst, const void * symbol, size_t count, size_t offset, cudaMemcpyKind kind); 
# 10063 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/cuda_runtime_api.h"
extern cudaError_t cudaGraphMemcpyNodeSetParams1D(cudaGraphNode_t node, void * dst, const void * src, size_t count, cudaMemcpyKind kind); 
# 10110 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/cuda_runtime_api.h"
extern cudaError_t cudaGraphAddMemsetNode(cudaGraphNode_t * pGraphNode, cudaGraph_t graph, const cudaGraphNode_t * pDependencies, size_t numDependencies, const cudaMemsetParams * pMemsetParams); 
# 10133 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/cuda_runtime_api.h"
extern cudaError_t cudaGraphMemsetNodeGetParams(cudaGraphNode_t node, cudaMemsetParams * pNodeParams); 
# 10156 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/cuda_runtime_api.h"
extern cudaError_t cudaGraphMemsetNodeSetParams(cudaGraphNode_t node, const cudaMemsetParams * pNodeParams); 
# 10197 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/cuda_runtime_api.h"
extern cudaError_t cudaGraphAddHostNode(cudaGraphNode_t * pGraphNode, cudaGraph_t graph, const cudaGraphNode_t * pDependencies, size_t numDependencies, const cudaHostNodeParams * pNodeParams); 
# 10220 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/cuda_runtime_api.h"
extern cudaError_t cudaGraphHostNodeGetParams(cudaGraphNode_t node, cudaHostNodeParams * pNodeParams); 
# 10243 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/cuda_runtime_api.h"
extern cudaError_t cudaGraphHostNodeSetParams(cudaGraphNode_t node, const cudaHostNodeParams * pNodeParams); 
# 10283 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/cuda_runtime_api.h"
extern cudaError_t cudaGraphAddChildGraphNode(cudaGraphNode_t * pGraphNode, cudaGraph_t graph, const cudaGraphNode_t * pDependencies, size_t numDependencies, cudaGraph_t childGraph); 
# 10310 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/cuda_runtime_api.h"
extern cudaError_t cudaGraphChildGraphNodeGetGraph(cudaGraphNode_t node, cudaGraph_t * pGraph); 
# 10347 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/cuda_runtime_api.h"
extern cudaError_t cudaGraphAddEmptyNode(cudaGraphNode_t * pGraphNode, cudaGraph_t graph, const cudaGraphNode_t * pDependencies, size_t numDependencies); 
# 10390 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/cuda_runtime_api.h"
extern cudaError_t cudaGraphAddEventRecordNode(cudaGraphNode_t * pGraphNode, cudaGraph_t graph, const cudaGraphNode_t * pDependencies, size_t numDependencies, cudaEvent_t event); 
# 10417 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/cuda_runtime_api.h"
extern cudaError_t cudaGraphEventRecordNodeGetEvent(cudaGraphNode_t node, cudaEvent_t * event_out); 
# 10444 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/cuda_runtime_api.h"
extern cudaError_t cudaGraphEventRecordNodeSetEvent(cudaGraphNode_t node, cudaEvent_t event); 
# 10490 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/cuda_runtime_api.h"
extern cudaError_t cudaGraphAddEventWaitNode(cudaGraphNode_t * pGraphNode, cudaGraph_t graph, const cudaGraphNode_t * pDependencies, size_t numDependencies, cudaEvent_t event); 
# 10517 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/cuda_runtime_api.h"
extern cudaError_t cudaGraphEventWaitNodeGetEvent(cudaGraphNode_t node, cudaEvent_t * event_out); 
# 10544 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/cuda_runtime_api.h"
extern cudaError_t cudaGraphEventWaitNodeSetEvent(cudaGraphNode_t node, cudaEvent_t event); 
# 10593 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/cuda_runtime_api.h"
extern cudaError_t cudaGraphAddExternalSemaphoresSignalNode(cudaGraphNode_t * pGraphNode, cudaGraph_t graph, const cudaGraphNode_t * pDependencies, size_t numDependencies, const cudaExternalSemaphoreSignalNodeParams * nodeParams); 
# 10626 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/cuda_runtime_api.h"
extern cudaError_t cudaGraphExternalSemaphoresSignalNodeGetParams(cudaGraphNode_t hNode, cudaExternalSemaphoreSignalNodeParams * params_out); 
# 10653 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/cuda_runtime_api.h"
extern cudaError_t cudaGraphExternalSemaphoresSignalNodeSetParams(cudaGraphNode_t hNode, const cudaExternalSemaphoreSignalNodeParams * nodeParams); 
# 10702 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/cuda_runtime_api.h"
extern cudaError_t cudaGraphAddExternalSemaphoresWaitNode(cudaGraphNode_t * pGraphNode, cudaGraph_t graph, const cudaGraphNode_t * pDependencies, size_t numDependencies, const cudaExternalSemaphoreWaitNodeParams * nodeParams); 
# 10735 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/cuda_runtime_api.h"
extern cudaError_t cudaGraphExternalSemaphoresWaitNodeGetParams(cudaGraphNode_t hNode, cudaExternalSemaphoreWaitNodeParams * params_out); 
# 10762 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/cuda_runtime_api.h"
extern cudaError_t cudaGraphExternalSemaphoresWaitNodeSetParams(cudaGraphNode_t hNode, const cudaExternalSemaphoreWaitNodeParams * nodeParams); 
# 10839 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/cuda_runtime_api.h"
extern cudaError_t cudaGraphAddMemAllocNode(cudaGraphNode_t * pGraphNode, cudaGraph_t graph, const cudaGraphNode_t * pDependencies, size_t numDependencies, cudaMemAllocNodeParams * nodeParams); 
# 10866 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/cuda_runtime_api.h"
extern cudaError_t cudaGraphMemAllocNodeGetParams(cudaGraphNode_t node, cudaMemAllocNodeParams * params_out); 
# 10926 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/cuda_runtime_api.h"
extern cudaError_t cudaGraphAddMemFreeNode(cudaGraphNode_t * pGraphNode, cudaGraph_t graph, const cudaGraphNode_t * pDependencies, size_t numDependencies, void * dptr); 
# 10950 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/cuda_runtime_api.h"
extern cudaError_t cudaGraphMemFreeNodeGetParams(cudaGraphNode_t node, void * dptr_out); 
# 10978 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/cuda_runtime_api.h"
extern cudaError_t cudaDeviceGraphMemTrim(int device); 
# 11015 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/cuda_runtime_api.h"
extern cudaError_t cudaDeviceGetGraphMemAttribute(int device, cudaGraphMemAttributeType attr, void * value); 
# 11049 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/cuda_runtime_api.h"
extern cudaError_t cudaDeviceSetGraphMemAttribute(int device, cudaGraphMemAttributeType attr, void * value); 
# 11077 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/cuda_runtime_api.h"
extern cudaError_t cudaGraphClone(cudaGraph_t * pGraphClone, cudaGraph_t originalGraph); 
# 11105 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/cuda_runtime_api.h"
extern cudaError_t cudaGraphNodeFindInClone(cudaGraphNode_t * pNode, cudaGraphNode_t originalNode, cudaGraph_t clonedGraph); 
# 11136 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/cuda_runtime_api.h"
extern cudaError_t cudaGraphNodeGetType(cudaGraphNode_t node, cudaGraphNodeType * pType); 
# 11167 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/cuda_runtime_api.h"
extern cudaError_t cudaGraphGetNodes(cudaGraph_t graph, cudaGraphNode_t * nodes, size_t * numNodes); 
# 11198 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/cuda_runtime_api.h"
extern cudaError_t cudaGraphGetRootNodes(cudaGraph_t graph, cudaGraphNode_t * pRootNodes, size_t * pNumRootNodes); 
# 11232 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/cuda_runtime_api.h"
extern cudaError_t cudaGraphGetEdges(cudaGraph_t graph, cudaGraphNode_t * from, cudaGraphNode_t * to, size_t * numEdges); 
# 11263 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/cuda_runtime_api.h"
extern cudaError_t cudaGraphNodeGetDependencies(cudaGraphNode_t node, cudaGraphNode_t * pDependencies, size_t * pNumDependencies); 
# 11295 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/cuda_runtime_api.h"
extern cudaError_t cudaGraphNodeGetDependentNodes(cudaGraphNode_t node, cudaGraphNode_t * pDependentNodes, size_t * pNumDependentNodes); 
# 11326 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/cuda_runtime_api.h"
extern cudaError_t cudaGraphAddDependencies(cudaGraph_t graph, const cudaGraphNode_t * from, const cudaGraphNode_t * to, size_t numDependencies); 
# 11357 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/cuda_runtime_api.h"
extern cudaError_t cudaGraphRemoveDependencies(cudaGraph_t graph, const cudaGraphNode_t * from, const cudaGraphNode_t * to, size_t numDependencies); 
# 11387 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/cuda_runtime_api.h"
extern cudaError_t cudaGraphDestroyNode(cudaGraphNode_t node); 
# 11425 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/cuda_runtime_api.h"
extern cudaError_t cudaGraphInstantiate(cudaGraphExec_t * pGraphExec, cudaGraph_t graph, cudaGraphNode_t * pErrorNode, char * pLogBuffer, size_t bufferSize); 
# 11468 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/cuda_runtime_api.h"
extern cudaError_t cudaGraphInstantiateWithFlags(cudaGraphExec_t * pGraphExec, cudaGraph_t graph, unsigned long long flags); 
# 11512 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/cuda_runtime_api.h"
extern cudaError_t cudaGraphExecKernelNodeSetParams(cudaGraphExec_t hGraphExec, cudaGraphNode_t node, const cudaKernelNodeParams * pNodeParams); 
# 11562 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/cuda_runtime_api.h"
extern cudaError_t cudaGraphExecMemcpyNodeSetParams(cudaGraphExec_t hGraphExec, cudaGraphNode_t node, const cudaMemcpy3DParms * pNodeParams); 
# 11617 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/cuda_runtime_api.h"
extern cudaError_t cudaGraphExecMemcpyNodeSetParamsToSymbol(cudaGraphExec_t hGraphExec, cudaGraphNode_t node, const void * symbol, const void * src, size_t count, size_t offset, cudaMemcpyKind kind); 
# 11680 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/cuda_runtime_api.h"
extern cudaError_t cudaGraphExecMemcpyNodeSetParamsFromSymbol(cudaGraphExec_t hGraphExec, cudaGraphNode_t node, void * dst, const void * symbol, size_t count, size_t offset, cudaMemcpyKind kind); 
# 11741 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/cuda_runtime_api.h"
extern cudaError_t cudaGraphExecMemcpyNodeSetParams1D(cudaGraphExec_t hGraphExec, cudaGraphNode_t node, void * dst, const void * src, size_t count, cudaMemcpyKind kind); 
# 11795 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/cuda_runtime_api.h"
extern cudaError_t cudaGraphExecMemsetNodeSetParams(cudaGraphExec_t hGraphExec, cudaGraphNode_t node, const cudaMemsetParams * pNodeParams); 
# 11834 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/cuda_runtime_api.h"
extern cudaError_t cudaGraphExecHostNodeSetParams(cudaGraphExec_t hGraphExec, cudaGraphNode_t node, const cudaHostNodeParams * pNodeParams); 
# 11880 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/cuda_runtime_api.h"
extern cudaError_t cudaGraphExecChildGraphNodeSetParams(cudaGraphExec_t hGraphExec, cudaGraphNode_t node, cudaGraph_t childGraph); 
# 11924 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/cuda_runtime_api.h"
extern cudaError_t cudaGraphExecEventRecordNodeSetEvent(cudaGraphExec_t hGraphExec, cudaGraphNode_t hNode, cudaEvent_t event); 
# 11968 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/cuda_runtime_api.h"
extern cudaError_t cudaGraphExecEventWaitNodeSetEvent(cudaGraphExec_t hGraphExec, cudaGraphNode_t hNode, cudaEvent_t event); 
# 12015 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/cuda_runtime_api.h"
extern cudaError_t cudaGraphExecExternalSemaphoresSignalNodeSetParams(cudaGraphExec_t hGraphExec, cudaGraphNode_t hNode, const cudaExternalSemaphoreSignalNodeParams * nodeParams); 
# 12062 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/cuda_runtime_api.h"
extern cudaError_t cudaGraphExecExternalSemaphoresWaitNodeSetParams(cudaGraphExec_t hGraphExec, cudaGraphNode_t hNode, const cudaExternalSemaphoreWaitNodeParams * nodeParams); 
# 12137 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/cuda_runtime_api.h"
extern cudaError_t cudaGraphExecUpdate(cudaGraphExec_t hGraphExec, cudaGraph_t hGraph, cudaGraphNode_t * hErrorNode_out, cudaGraphExecUpdateResult * updateResult_out); 
# 12162 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/cuda_runtime_api.h"
extern cudaError_t cudaGraphUpload(cudaGraphExec_t graphExec, cudaStream_t stream); 
# 12193 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/cuda_runtime_api.h"
extern cudaError_t cudaGraphLaunch(cudaGraphExec_t graphExec, cudaStream_t stream); 
# 12216 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/cuda_runtime_api.h"
extern cudaError_t cudaGraphExecDestroy(cudaGraphExec_t graphExec); 
# 12237 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/cuda_runtime_api.h"
extern cudaError_t cudaGraphDestroy(cudaGraph_t graph); 
# 12256 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/cuda_runtime_api.h"
extern cudaError_t cudaGraphDebugDotPrint(cudaGraph_t graph, const char * path, unsigned flags); 
# 12292 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/cuda_runtime_api.h"
extern cudaError_t cudaUserObjectCreate(cudaUserObject_t * object_out, void * ptr, cudaHostFn_t destroy, unsigned initialRefcount, unsigned flags); 
# 12316 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/cuda_runtime_api.h"
extern cudaError_t cudaUserObjectRetain(cudaUserObject_t object, unsigned count = 1); 
# 12344 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/cuda_runtime_api.h"
extern cudaError_t cudaUserObjectRelease(cudaUserObject_t object, unsigned count = 1); 
# 12372 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/cuda_runtime_api.h"
extern cudaError_t cudaGraphRetainUserObject(cudaGraph_t graph, cudaUserObject_t object, unsigned count = 1, unsigned flags = 0); 
# 12397 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/cuda_runtime_api.h"
extern cudaError_t cudaGraphReleaseUserObject(cudaGraph_t graph, cudaUserObject_t object, unsigned count = 1); 
# 12463 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/cuda_runtime_api.h"
extern cudaError_t cudaGetDriverEntryPoint(const char * symbol, void ** funcPtr, unsigned long long flags); 
# 12468
extern cudaError_t cudaGetExportTable(const void ** ppExportTable, const cudaUUID_t * pExportTableId); 
# 12644 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/cuda_runtime_api.h"
extern cudaError_t cudaGetFuncBySymbol(cudaFunction_t * functionPtr, const void * symbolPtr); 
# 12786 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/cuda_runtime_api.h"
}
# 104 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/channel_descriptor.h"
template< class T> inline cudaChannelFormatDesc cudaCreateChannelDesc() 
# 105
{ 
# 106
return cudaCreateChannelDesc(0, 0, 0, 0, cudaChannelFormatKindNone); 
# 107
} 
# 109
static inline cudaChannelFormatDesc cudaCreateChannelDescHalf() 
# 110
{ 
# 111
int e = (((int)sizeof(unsigned short)) * 8); 
# 113
return cudaCreateChannelDesc(e, 0, 0, 0, cudaChannelFormatKindFloat); 
# 114
} 
# 116
static inline cudaChannelFormatDesc cudaCreateChannelDescHalf1() 
# 117
{ 
# 118
int e = (((int)sizeof(unsigned short)) * 8); 
# 120
return cudaCreateChannelDesc(e, 0, 0, 0, cudaChannelFormatKindFloat); 
# 121
} 
# 123
static inline cudaChannelFormatDesc cudaCreateChannelDescHalf2() 
# 124
{ 
# 125
int e = (((int)sizeof(unsigned short)) * 8); 
# 127
return cudaCreateChannelDesc(e, e, 0, 0, cudaChannelFormatKindFloat); 
# 128
} 
# 130
static inline cudaChannelFormatDesc cudaCreateChannelDescHalf4() 
# 131
{ 
# 132
int e = (((int)sizeof(unsigned short)) * 8); 
# 134
return cudaCreateChannelDesc(e, e, e, e, cudaChannelFormatKindFloat); 
# 135
} 
# 137
template<> inline cudaChannelFormatDesc cudaCreateChannelDesc< char> () 
# 138
{ 
# 139
int e = (((int)sizeof(char)) * 8); 
# 144
return cudaCreateChannelDesc(e, 0, 0, 0, cudaChannelFormatKindSigned); 
# 146
} 
# 148
template<> inline cudaChannelFormatDesc cudaCreateChannelDesc< signed char> () 
# 149
{ 
# 150
int e = (((int)sizeof(signed char)) * 8); 
# 152
return cudaCreateChannelDesc(e, 0, 0, 0, cudaChannelFormatKindSigned); 
# 153
} 
# 155
template<> inline cudaChannelFormatDesc cudaCreateChannelDesc< unsigned char> () 
# 156
{ 
# 157
int e = (((int)sizeof(unsigned char)) * 8); 
# 159
return cudaCreateChannelDesc(e, 0, 0, 0, cudaChannelFormatKindUnsigned); 
# 160
} 
# 162
template<> inline cudaChannelFormatDesc cudaCreateChannelDesc< char1> () 
# 163
{ 
# 164
int e = (((int)sizeof(signed char)) * 8); 
# 166
return cudaCreateChannelDesc(e, 0, 0, 0, cudaChannelFormatKindSigned); 
# 167
} 
# 169
template<> inline cudaChannelFormatDesc cudaCreateChannelDesc< uchar1> () 
# 170
{ 
# 171
int e = (((int)sizeof(unsigned char)) * 8); 
# 173
return cudaCreateChannelDesc(e, 0, 0, 0, cudaChannelFormatKindUnsigned); 
# 174
} 
# 176
template<> inline cudaChannelFormatDesc cudaCreateChannelDesc< char2> () 
# 177
{ 
# 178
int e = (((int)sizeof(signed char)) * 8); 
# 180
return cudaCreateChannelDesc(e, e, 0, 0, cudaChannelFormatKindSigned); 
# 181
} 
# 183
template<> inline cudaChannelFormatDesc cudaCreateChannelDesc< uchar2> () 
# 184
{ 
# 185
int e = (((int)sizeof(unsigned char)) * 8); 
# 187
return cudaCreateChannelDesc(e, e, 0, 0, cudaChannelFormatKindUnsigned); 
# 188
} 
# 190
template<> inline cudaChannelFormatDesc cudaCreateChannelDesc< char4> () 
# 191
{ 
# 192
int e = (((int)sizeof(signed char)) * 8); 
# 194
return cudaCreateChannelDesc(e, e, e, e, cudaChannelFormatKindSigned); 
# 195
} 
# 197
template<> inline cudaChannelFormatDesc cudaCreateChannelDesc< uchar4> () 
# 198
{ 
# 199
int e = (((int)sizeof(unsigned char)) * 8); 
# 201
return cudaCreateChannelDesc(e, e, e, e, cudaChannelFormatKindUnsigned); 
# 202
} 
# 204
template<> inline cudaChannelFormatDesc cudaCreateChannelDesc< short> () 
# 205
{ 
# 206
int e = (((int)sizeof(short)) * 8); 
# 208
return cudaCreateChannelDesc(e, 0, 0, 0, cudaChannelFormatKindSigned); 
# 209
} 
# 211
template<> inline cudaChannelFormatDesc cudaCreateChannelDesc< unsigned short> () 
# 212
{ 
# 213
int e = (((int)sizeof(unsigned short)) * 8); 
# 215
return cudaCreateChannelDesc(e, 0, 0, 0, cudaChannelFormatKindUnsigned); 
# 216
} 
# 218
template<> inline cudaChannelFormatDesc cudaCreateChannelDesc< short1> () 
# 219
{ 
# 220
int e = (((int)sizeof(short)) * 8); 
# 222
return cudaCreateChannelDesc(e, 0, 0, 0, cudaChannelFormatKindSigned); 
# 223
} 
# 225
template<> inline cudaChannelFormatDesc cudaCreateChannelDesc< ushort1> () 
# 226
{ 
# 227
int e = (((int)sizeof(unsigned short)) * 8); 
# 229
return cudaCreateChannelDesc(e, 0, 0, 0, cudaChannelFormatKindUnsigned); 
# 230
} 
# 232
template<> inline cudaChannelFormatDesc cudaCreateChannelDesc< short2> () 
# 233
{ 
# 234
int e = (((int)sizeof(short)) * 8); 
# 236
return cudaCreateChannelDesc(e, e, 0, 0, cudaChannelFormatKindSigned); 
# 237
} 
# 239
template<> inline cudaChannelFormatDesc cudaCreateChannelDesc< ushort2> () 
# 240
{ 
# 241
int e = (((int)sizeof(unsigned short)) * 8); 
# 243
return cudaCreateChannelDesc(e, e, 0, 0, cudaChannelFormatKindUnsigned); 
# 244
} 
# 246
template<> inline cudaChannelFormatDesc cudaCreateChannelDesc< short4> () 
# 247
{ 
# 248
int e = (((int)sizeof(short)) * 8); 
# 250
return cudaCreateChannelDesc(e, e, e, e, cudaChannelFormatKindSigned); 
# 251
} 
# 253
template<> inline cudaChannelFormatDesc cudaCreateChannelDesc< ushort4> () 
# 254
{ 
# 255
int e = (((int)sizeof(unsigned short)) * 8); 
# 257
return cudaCreateChannelDesc(e, e, e, e, cudaChannelFormatKindUnsigned); 
# 258
} 
# 260
template<> inline cudaChannelFormatDesc cudaCreateChannelDesc< int> () 
# 261
{ 
# 262
int e = (((int)sizeof(int)) * 8); 
# 264
return cudaCreateChannelDesc(e, 0, 0, 0, cudaChannelFormatKindSigned); 
# 265
} 
# 267
template<> inline cudaChannelFormatDesc cudaCreateChannelDesc< unsigned> () 
# 268
{ 
# 269
int e = (((int)sizeof(unsigned)) * 8); 
# 271
return cudaCreateChannelDesc(e, 0, 0, 0, cudaChannelFormatKindUnsigned); 
# 272
} 
# 274
template<> inline cudaChannelFormatDesc cudaCreateChannelDesc< int1> () 
# 275
{ 
# 276
int e = (((int)sizeof(int)) * 8); 
# 278
return cudaCreateChannelDesc(e, 0, 0, 0, cudaChannelFormatKindSigned); 
# 279
} 
# 281
template<> inline cudaChannelFormatDesc cudaCreateChannelDesc< uint1> () 
# 282
{ 
# 283
int e = (((int)sizeof(unsigned)) * 8); 
# 285
return cudaCreateChannelDesc(e, 0, 0, 0, cudaChannelFormatKindUnsigned); 
# 286
} 
# 288
template<> inline cudaChannelFormatDesc cudaCreateChannelDesc< int2> () 
# 289
{ 
# 290
int e = (((int)sizeof(int)) * 8); 
# 292
return cudaCreateChannelDesc(e, e, 0, 0, cudaChannelFormatKindSigned); 
# 293
} 
# 295
template<> inline cudaChannelFormatDesc cudaCreateChannelDesc< uint2> () 
# 296
{ 
# 297
int e = (((int)sizeof(unsigned)) * 8); 
# 299
return cudaCreateChannelDesc(e, e, 0, 0, cudaChannelFormatKindUnsigned); 
# 300
} 
# 302
template<> inline cudaChannelFormatDesc cudaCreateChannelDesc< int4> () 
# 303
{ 
# 304
int e = (((int)sizeof(int)) * 8); 
# 306
return cudaCreateChannelDesc(e, e, e, e, cudaChannelFormatKindSigned); 
# 307
} 
# 309
template<> inline cudaChannelFormatDesc cudaCreateChannelDesc< uint4> () 
# 310
{ 
# 311
int e = (((int)sizeof(unsigned)) * 8); 
# 313
return cudaCreateChannelDesc(e, e, e, e, cudaChannelFormatKindUnsigned); 
# 314
} 
# 376 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/channel_descriptor.h"
template<> inline cudaChannelFormatDesc cudaCreateChannelDesc< float> () 
# 377
{ 
# 378
int e = (((int)sizeof(float)) * 8); 
# 380
return cudaCreateChannelDesc(e, 0, 0, 0, cudaChannelFormatKindFloat); 
# 381
} 
# 383
template<> inline cudaChannelFormatDesc cudaCreateChannelDesc< float1> () 
# 384
{ 
# 385
int e = (((int)sizeof(float)) * 8); 
# 387
return cudaCreateChannelDesc(e, 0, 0, 0, cudaChannelFormatKindFloat); 
# 388
} 
# 390
template<> inline cudaChannelFormatDesc cudaCreateChannelDesc< float2> () 
# 391
{ 
# 392
int e = (((int)sizeof(float)) * 8); 
# 394
return cudaCreateChannelDesc(e, e, 0, 0, cudaChannelFormatKindFloat); 
# 395
} 
# 397
template<> inline cudaChannelFormatDesc cudaCreateChannelDesc< float4> () 
# 398
{ 
# 399
int e = (((int)sizeof(float)) * 8); 
# 401
return cudaCreateChannelDesc(e, e, e, e, cudaChannelFormatKindFloat); 
# 402
} 
# 404
static inline cudaChannelFormatDesc cudaCreateChannelDescNV12() 
# 405
{ 
# 406
int e = (((int)sizeof(char)) * 8); 
# 408
return cudaCreateChannelDesc(e, e, e, 0, cudaChannelFormatKindNV12); 
# 409
} 
# 79 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/driver_functions.h"
static inline cudaPitchedPtr make_cudaPitchedPtr(void *d, size_t p, size_t xsz, size_t ysz) 
# 80
{ 
# 81
cudaPitchedPtr s; 
# 83
(s.ptr) = d; 
# 84
(s.pitch) = p; 
# 85
(s.xsize) = xsz; 
# 86
(s.ysize) = ysz; 
# 88
return s; 
# 89
} 
# 106 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/driver_functions.h"
static inline cudaPos make_cudaPos(size_t x, size_t y, size_t z) 
# 107
{ 
# 108
cudaPos p; 
# 110
(p.x) = x; 
# 111
(p.y) = y; 
# 112
(p.z) = z; 
# 114
return p; 
# 115
} 
# 132 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/driver_functions.h"
static inline cudaExtent make_cudaExtent(size_t w, size_t h, size_t d) 
# 133
{ 
# 134
cudaExtent e; 
# 136
(e.width) = w; 
# 137
(e.height) = h; 
# 138
(e.depth) = d; 
# 140
return e; 
# 141
} 
# 73 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/vector_functions.h"
static inline char1 make_char1(signed char x); 
# 75
static inline uchar1 make_uchar1(unsigned char x); 
# 77
static inline char2 make_char2(signed char x, signed char y); 
# 79
static inline uchar2 make_uchar2(unsigned char x, unsigned char y); 
# 81
static inline char3 make_char3(signed char x, signed char y, signed char z); 
# 83
static inline uchar3 make_uchar3(unsigned char x, unsigned char y, unsigned char z); 
# 85
static inline char4 make_char4(signed char x, signed char y, signed char z, signed char w); 
# 87
static inline uchar4 make_uchar4(unsigned char x, unsigned char y, unsigned char z, unsigned char w); 
# 89
static inline short1 make_short1(short x); 
# 91
static inline ushort1 make_ushort1(unsigned short x); 
# 93
static inline short2 make_short2(short x, short y); 
# 95
static inline ushort2 make_ushort2(unsigned short x, unsigned short y); 
# 97
static inline short3 make_short3(short x, short y, short z); 
# 99
static inline ushort3 make_ushort3(unsigned short x, unsigned short y, unsigned short z); 
# 101
static inline short4 make_short4(short x, short y, short z, short w); 
# 103
static inline ushort4 make_ushort4(unsigned short x, unsigned short y, unsigned short z, unsigned short w); 
# 105
static inline int1 make_int1(int x); 
# 107
static inline uint1 make_uint1(unsigned x); 
# 109
static inline int2 make_int2(int x, int y); 
# 111
static inline uint2 make_uint2(unsigned x, unsigned y); 
# 113
static inline int3 make_int3(int x, int y, int z); 
# 115
static inline uint3 make_uint3(unsigned x, unsigned y, unsigned z); 
# 117
static inline int4 make_int4(int x, int y, int z, int w); 
# 119
static inline uint4 make_uint4(unsigned x, unsigned y, unsigned z, unsigned w); 
# 121
static inline long1 make_long1(long x); 
# 123
static inline ulong1 make_ulong1(unsigned long x); 
# 125
static inline long2 make_long2(long x, long y); 
# 127
static inline ulong2 make_ulong2(unsigned long x, unsigned long y); 
# 129
static inline long3 make_long3(long x, long y, long z); 
# 131
static inline ulong3 make_ulong3(unsigned long x, unsigned long y, unsigned long z); 
# 133
static inline long4 make_long4(long x, long y, long z, long w); 
# 135
static inline ulong4 make_ulong4(unsigned long x, unsigned long y, unsigned long z, unsigned long w); 
# 137
static inline float1 make_float1(float x); 
# 139
static inline float2 make_float2(float x, float y); 
# 141
static inline float3 make_float3(float x, float y, float z); 
# 143
static inline float4 make_float4(float x, float y, float z, float w); 
# 145
static inline longlong1 make_longlong1(long long x); 
# 147
static inline ulonglong1 make_ulonglong1(unsigned long long x); 
# 149
static inline longlong2 make_longlong2(long long x, long long y); 
# 151
static inline ulonglong2 make_ulonglong2(unsigned long long x, unsigned long long y); 
# 153
static inline longlong3 make_longlong3(long long x, long long y, long long z); 
# 155
static inline ulonglong3 make_ulonglong3(unsigned long long x, unsigned long long y, unsigned long long z); 
# 157
static inline longlong4 make_longlong4(long long x, long long y, long long z, long long w); 
# 159
static inline ulonglong4 make_ulonglong4(unsigned long long x, unsigned long long y, unsigned long long z, unsigned long long w); 
# 161
static inline double1 make_double1(double x); 
# 163
static inline double2 make_double2(double x, double y); 
# 165
static inline double3 make_double3(double x, double y, double z); 
# 167
static inline double4 make_double4(double x, double y, double z, double w); 
# 73 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/vector_functions.hpp"
static inline char1 make_char1(signed char x) 
# 74
{ 
# 75
char1 t; (t.x) = x; return t; 
# 76
} 
# 78
static inline uchar1 make_uchar1(unsigned char x) 
# 79
{ 
# 80
uchar1 t; (t.x) = x; return t; 
# 81
} 
# 83
static inline char2 make_char2(signed char x, signed char y) 
# 84
{ 
# 85
char2 t; (t.x) = x; (t.y) = y; return t; 
# 86
} 
# 88
static inline uchar2 make_uchar2(unsigned char x, unsigned char y) 
# 89
{ 
# 90
uchar2 t; (t.x) = x; (t.y) = y; return t; 
# 91
} 
# 93
static inline char3 make_char3(signed char x, signed char y, signed char z) 
# 94
{ 
# 95
char3 t; (t.x) = x; (t.y) = y; (t.z) = z; return t; 
# 96
} 
# 98
static inline uchar3 make_uchar3(unsigned char x, unsigned char y, unsigned char z) 
# 99
{ 
# 100
uchar3 t; (t.x) = x; (t.y) = y; (t.z) = z; return t; 
# 101
} 
# 103
static inline char4 make_char4(signed char x, signed char y, signed char z, signed char w) 
# 104
{ 
# 105
char4 t; (t.x) = x; (t.y) = y; (t.z) = z; (t.w) = w; return t; 
# 106
} 
# 108
static inline uchar4 make_uchar4(unsigned char x, unsigned char y, unsigned char z, unsigned char w) 
# 109
{ 
# 110
uchar4 t; (t.x) = x; (t.y) = y; (t.z) = z; (t.w) = w; return t; 
# 111
} 
# 113
static inline short1 make_short1(short x) 
# 114
{ 
# 115
short1 t; (t.x) = x; return t; 
# 116
} 
# 118
static inline ushort1 make_ushort1(unsigned short x) 
# 119
{ 
# 120
ushort1 t; (t.x) = x; return t; 
# 121
} 
# 123
static inline short2 make_short2(short x, short y) 
# 124
{ 
# 125
short2 t; (t.x) = x; (t.y) = y; return t; 
# 126
} 
# 128
static inline ushort2 make_ushort2(unsigned short x, unsigned short y) 
# 129
{ 
# 130
ushort2 t; (t.x) = x; (t.y) = y; return t; 
# 131
} 
# 133
static inline short3 make_short3(short x, short y, short z) 
# 134
{ 
# 135
short3 t; (t.x) = x; (t.y) = y; (t.z) = z; return t; 
# 136
} 
# 138
static inline ushort3 make_ushort3(unsigned short x, unsigned short y, unsigned short z) 
# 139
{ 
# 140
ushort3 t; (t.x) = x; (t.y) = y; (t.z) = z; return t; 
# 141
} 
# 143
static inline short4 make_short4(short x, short y, short z, short w) 
# 144
{ 
# 145
short4 t; (t.x) = x; (t.y) = y; (t.z) = z; (t.w) = w; return t; 
# 146
} 
# 148
static inline ushort4 make_ushort4(unsigned short x, unsigned short y, unsigned short z, unsigned short w) 
# 149
{ 
# 150
ushort4 t; (t.x) = x; (t.y) = y; (t.z) = z; (t.w) = w; return t; 
# 151
} 
# 153
static inline int1 make_int1(int x) 
# 154
{ 
# 155
int1 t; (t.x) = x; return t; 
# 156
} 
# 158
static inline uint1 make_uint1(unsigned x) 
# 159
{ 
# 160
uint1 t; (t.x) = x; return t; 
# 161
} 
# 163
static inline int2 make_int2(int x, int y) 
# 164
{ 
# 165
int2 t; (t.x) = x; (t.y) = y; return t; 
# 166
} 
# 168
static inline uint2 make_uint2(unsigned x, unsigned y) 
# 169
{ 
# 170
uint2 t; (t.x) = x; (t.y) = y; return t; 
# 171
} 
# 173
static inline int3 make_int3(int x, int y, int z) 
# 174
{ 
# 175
int3 t; (t.x) = x; (t.y) = y; (t.z) = z; return t; 
# 176
} 
# 178
static inline uint3 make_uint3(unsigned x, unsigned y, unsigned z) 
# 179
{ 
# 180
uint3 t; (t.x) = x; (t.y) = y; (t.z) = z; return t; 
# 181
} 
# 183
static inline int4 make_int4(int x, int y, int z, int w) 
# 184
{ 
# 185
int4 t; (t.x) = x; (t.y) = y; (t.z) = z; (t.w) = w; return t; 
# 186
} 
# 188
static inline uint4 make_uint4(unsigned x, unsigned y, unsigned z, unsigned w) 
# 189
{ 
# 190
uint4 t; (t.x) = x; (t.y) = y; (t.z) = z; (t.w) = w; return t; 
# 191
} 
# 193
static inline long1 make_long1(long x) 
# 194
{ 
# 195
long1 t; (t.x) = x; return t; 
# 196
} 
# 198
static inline ulong1 make_ulong1(unsigned long x) 
# 199
{ 
# 200
ulong1 t; (t.x) = x; return t; 
# 201
} 
# 203
static inline long2 make_long2(long x, long y) 
# 204
{ 
# 205
long2 t; (t.x) = x; (t.y) = y; return t; 
# 206
} 
# 208
static inline ulong2 make_ulong2(unsigned long x, unsigned long y) 
# 209
{ 
# 210
ulong2 t; (t.x) = x; (t.y) = y; return t; 
# 211
} 
# 213
static inline long3 make_long3(long x, long y, long z) 
# 214
{ 
# 215
long3 t; (t.x) = x; (t.y) = y; (t.z) = z; return t; 
# 216
} 
# 218
static inline ulong3 make_ulong3(unsigned long x, unsigned long y, unsigned long z) 
# 219
{ 
# 220
ulong3 t; (t.x) = x; (t.y) = y; (t.z) = z; return t; 
# 221
} 
# 223
static inline long4 make_long4(long x, long y, long z, long w) 
# 224
{ 
# 225
long4 t; (t.x) = x; (t.y) = y; (t.z) = z; (t.w) = w; return t; 
# 226
} 
# 228
static inline ulong4 make_ulong4(unsigned long x, unsigned long y, unsigned long z, unsigned long w) 
# 229
{ 
# 230
ulong4 t; (t.x) = x; (t.y) = y; (t.z) = z; (t.w) = w; return t; 
# 231
} 
# 233
static inline float1 make_float1(float x) 
# 234
{ 
# 235
float1 t; (t.x) = x; return t; 
# 236
} 
# 238
static inline float2 make_float2(float x, float y) 
# 239
{ 
# 240
float2 t; (t.x) = x; (t.y) = y; return t; 
# 241
} 
# 243
static inline float3 make_float3(float x, float y, float z) 
# 244
{ 
# 245
float3 t; (t.x) = x; (t.y) = y; (t.z) = z; return t; 
# 246
} 
# 248
static inline float4 make_float4(float x, float y, float z, float w) 
# 249
{ 
# 250
float4 t; (t.x) = x; (t.y) = y; (t.z) = z; (t.w) = w; return t; 
# 251
} 
# 253
static inline longlong1 make_longlong1(long long x) 
# 254
{ 
# 255
longlong1 t; (t.x) = x; return t; 
# 256
} 
# 258
static inline ulonglong1 make_ulonglong1(unsigned long long x) 
# 259
{ 
# 260
ulonglong1 t; (t.x) = x; return t; 
# 261
} 
# 263
static inline longlong2 make_longlong2(long long x, long long y) 
# 264
{ 
# 265
longlong2 t; (t.x) = x; (t.y) = y; return t; 
# 266
} 
# 268
static inline ulonglong2 make_ulonglong2(unsigned long long x, unsigned long long y) 
# 269
{ 
# 270
ulonglong2 t; (t.x) = x; (t.y) = y; return t; 
# 271
} 
# 273
static inline longlong3 make_longlong3(long long x, long long y, long long z) 
# 274
{ 
# 275
longlong3 t; (t.x) = x; (t.y) = y; (t.z) = z; return t; 
# 276
} 
# 278
static inline ulonglong3 make_ulonglong3(unsigned long long x, unsigned long long y, unsigned long long z) 
# 279
{ 
# 280
ulonglong3 t; (t.x) = x; (t.y) = y; (t.z) = z; return t; 
# 281
} 
# 283
static inline longlong4 make_longlong4(long long x, long long y, long long z, long long w) 
# 284
{ 
# 285
longlong4 t; (t.x) = x; (t.y) = y; (t.z) = z; (t.w) = w; return t; 
# 286
} 
# 288
static inline ulonglong4 make_ulonglong4(unsigned long long x, unsigned long long y, unsigned long long z, unsigned long long w) 
# 289
{ 
# 290
ulonglong4 t; (t.x) = x; (t.y) = y; (t.z) = z; (t.w) = w; return t; 
# 291
} 
# 293
static inline double1 make_double1(double x) 
# 294
{ 
# 295
double1 t; (t.x) = x; return t; 
# 296
} 
# 298
static inline double2 make_double2(double x, double y) 
# 299
{ 
# 300
double2 t; (t.x) = x; (t.y) = y; return t; 
# 301
} 
# 303
static inline double3 make_double3(double x, double y, double z) 
# 304
{ 
# 305
double3 t; (t.x) = x; (t.y) = y; (t.z) = z; return t; 
# 306
} 
# 308
static inline double4 make_double4(double x, double y, double z, double w) 
# 309
{ 
# 310
double4 t; (t.x) = x; (t.y) = y; (t.z) = z; (t.w) = w; return t; 
# 311
} 
# 27 "/usr/include/string.h" 3
extern "C" {
# 42 "/usr/include/string.h" 3
extern void *memcpy(void *__restrict__ __dest, const void *__restrict__ __src, size_t __n) throw()
# 43
 __attribute((__nonnull__(1, 2))); 
# 46
extern void *memmove(void * __dest, const void * __src, size_t __n) throw()
# 47
 __attribute((__nonnull__(1, 2))); 
# 54
extern void *memccpy(void *__restrict__ __dest, const void *__restrict__ __src, int __c, size_t __n) throw()
# 56
 __attribute((__nonnull__(1, 2))); 
# 62
extern void *memset(void * __s, int __c, size_t __n) throw() __attribute((__nonnull__(1))); 
# 65
extern int memcmp(const void * __s1, const void * __s2, size_t __n) throw()
# 66
 __attribute((__pure__)) __attribute((__nonnull__(1, 2))); 
# 70
extern "C++" {
# 72
extern void *memchr(void * __s, int __c, size_t __n) throw() __asm__("memchr")
# 73
 __attribute((__pure__)) __attribute((__nonnull__(1))); 
# 74
extern const void *memchr(const void * __s, int __c, size_t __n) throw() __asm__("memchr")
# 75
 __attribute((__pure__)) __attribute((__nonnull__(1))); 
# 90 "/usr/include/string.h" 3
}
# 101
extern "C++" void *rawmemchr(void * __s, int __c) throw() __asm__("rawmemchr")
# 102
 __attribute((__pure__)) __attribute((__nonnull__(1))); 
# 103
extern "C++" const void *rawmemchr(const void * __s, int __c) throw() __asm__("rawmemchr")
# 104
 __attribute((__pure__)) __attribute((__nonnull__(1))); 
# 112
extern "C++" void *memrchr(void * __s, int __c, size_t __n) throw() __asm__("memrchr")
# 113
 __attribute((__pure__)) __attribute((__nonnull__(1))); 
# 114
extern "C++" const void *memrchr(const void * __s, int __c, size_t __n) throw() __asm__("memrchr")
# 115
 __attribute((__pure__)) __attribute((__nonnull__(1))); 
# 125
extern char *strcpy(char *__restrict__ __dest, const char *__restrict__ __src) throw()
# 126
 __attribute((__nonnull__(1, 2))); 
# 128
extern char *strncpy(char *__restrict__ __dest, const char *__restrict__ __src, size_t __n) throw()
# 130
 __attribute((__nonnull__(1, 2))); 
# 133
extern char *strcat(char *__restrict__ __dest, const char *__restrict__ __src) throw()
# 134
 __attribute((__nonnull__(1, 2))); 
# 136
extern char *strncat(char *__restrict__ __dest, const char *__restrict__ __src, size_t __n) throw()
# 137
 __attribute((__nonnull__(1, 2))); 
# 140
extern int strcmp(const char * __s1, const char * __s2) throw()
# 141
 __attribute((__pure__)) __attribute((__nonnull__(1, 2))); 
# 143
extern int strncmp(const char * __s1, const char * __s2, size_t __n) throw()
# 144
 __attribute((__pure__)) __attribute((__nonnull__(1, 2))); 
# 147
extern int strcoll(const char * __s1, const char * __s2) throw()
# 148
 __attribute((__pure__)) __attribute((__nonnull__(1, 2))); 
# 150
extern size_t strxfrm(char *__restrict__ __dest, const char *__restrict__ __src, size_t __n) throw()
# 152
 __attribute((__nonnull__(2))); 
# 39 "/usr/include/xlocale.h" 3
typedef 
# 27
struct __locale_struct { 
# 30
struct __locale_data *__locales[13]; 
# 33
const unsigned short *__ctype_b; 
# 34
const int *__ctype_tolower; 
# 35
const int *__ctype_toupper; 
# 38
const char *__names[13]; 
# 39
} *__locale_t; 
# 42
typedef __locale_t locale_t; 
# 162 "/usr/include/string.h" 3
extern int strcoll_l(const char * __s1, const char * __s2, __locale_t __l) throw()
# 163
 __attribute((__pure__)) __attribute((__nonnull__(1, 2, 3))); 
# 165
extern size_t strxfrm_l(char * __dest, const char * __src, size_t __n, __locale_t __l) throw()
# 166
 __attribute((__nonnull__(2, 4))); 
# 172
extern char *strdup(const char * __s) throw()
# 173
 __attribute((__malloc__)) __attribute((__nonnull__(1))); 
# 180
extern char *strndup(const char * __string, size_t __n) throw()
# 181
 __attribute((__malloc__)) __attribute((__nonnull__(1))); 
# 210 "/usr/include/string.h" 3
extern "C++" {
# 212
extern char *strchr(char * __s, int __c) throw() __asm__("strchr")
# 213
 __attribute((__pure__)) __attribute((__nonnull__(1))); 
# 214
extern const char *strchr(const char * __s, int __c) throw() __asm__("strchr")
# 215
 __attribute((__pure__)) __attribute((__nonnull__(1))); 
# 230 "/usr/include/string.h" 3
}
# 237
extern "C++" {
# 239
extern char *strrchr(char * __s, int __c) throw() __asm__("strrchr")
# 240
 __attribute((__pure__)) __attribute((__nonnull__(1))); 
# 241
extern const char *strrchr(const char * __s, int __c) throw() __asm__("strrchr")
# 242
 __attribute((__pure__)) __attribute((__nonnull__(1))); 
# 257 "/usr/include/string.h" 3
}
# 268
extern "C++" char *strchrnul(char * __s, int __c) throw() __asm__("strchrnul")
# 269
 __attribute((__pure__)) __attribute((__nonnull__(1))); 
# 270
extern "C++" const char *strchrnul(const char * __s, int __c) throw() __asm__("strchrnul")
# 271
 __attribute((__pure__)) __attribute((__nonnull__(1))); 
# 281
extern size_t strcspn(const char * __s, const char * __reject) throw()
# 282
 __attribute((__pure__)) __attribute((__nonnull__(1, 2))); 
# 285
extern size_t strspn(const char * __s, const char * __accept) throw()
# 286
 __attribute((__pure__)) __attribute((__nonnull__(1, 2))); 
# 289
extern "C++" {
# 291
extern char *strpbrk(char * __s, const char * __accept) throw() __asm__("strpbrk")
# 292
 __attribute((__pure__)) __attribute((__nonnull__(1, 2))); 
# 293
extern const char *strpbrk(const char * __s, const char * __accept) throw() __asm__("strpbrk")
# 294
 __attribute((__pure__)) __attribute((__nonnull__(1, 2))); 
# 309 "/usr/include/string.h" 3
}
# 316
extern "C++" {
# 318
extern char *strstr(char * __haystack, const char * __needle) throw() __asm__("strstr")
# 319
 __attribute((__pure__)) __attribute((__nonnull__(1, 2))); 
# 320
extern const char *strstr(const char * __haystack, const char * __needle) throw() __asm__("strstr")
# 321
 __attribute((__pure__)) __attribute((__nonnull__(1, 2))); 
# 336 "/usr/include/string.h" 3
}
# 344
extern char *strtok(char *__restrict__ __s, const char *__restrict__ __delim) throw()
# 345
 __attribute((__nonnull__(2))); 
# 350
extern char *__strtok_r(char *__restrict__ __s, const char *__restrict__ __delim, char **__restrict__ __save_ptr) throw()
# 353
 __attribute((__nonnull__(2, 3))); 
# 355
extern char *strtok_r(char *__restrict__ __s, const char *__restrict__ __delim, char **__restrict__ __save_ptr) throw()
# 357
 __attribute((__nonnull__(2, 3))); 
# 363
extern "C++" char *strcasestr(char * __haystack, const char * __needle) throw() __asm__("strcasestr")
# 364
 __attribute((__pure__)) __attribute((__nonnull__(1, 2))); 
# 365
extern "C++" const char *strcasestr(const char * __haystack, const char * __needle) throw() __asm__("strcasestr")
# 367
 __attribute((__pure__)) __attribute((__nonnull__(1, 2))); 
# 378 "/usr/include/string.h" 3
extern void *memmem(const void * __haystack, size_t __haystacklen, const void * __needle, size_t __needlelen) throw()
# 380
 __attribute((__pure__)) __attribute((__nonnull__(1, 3))); 
# 384
extern void *__mempcpy(void *__restrict__ __dest, const void *__restrict__ __src, size_t __n) throw()
# 386
 __attribute((__nonnull__(1, 2))); 
# 387
extern void *mempcpy(void *__restrict__ __dest, const void *__restrict__ __src, size_t __n) throw()
# 389
 __attribute((__nonnull__(1, 2))); 
# 395
extern size_t strlen(const char * __s) throw()
# 396
 __attribute((__pure__)) __attribute((__nonnull__(1))); 
# 402
extern size_t strnlen(const char * __string, size_t __maxlen) throw()
# 403
 __attribute((__pure__)) __attribute((__nonnull__(1))); 
# 409
extern char *strerror(int __errnum) throw(); 
# 434 "/usr/include/string.h" 3
extern char *strerror_r(int __errnum, char * __buf, size_t __buflen) throw()
# 435
 __attribute((__nonnull__(2))); 
# 441
extern char *strerror_l(int __errnum, __locale_t __l) throw(); 
# 447
extern void __bzero(void * __s, size_t __n) throw() __attribute((__nonnull__(1))); 
# 451
extern void bcopy(const void * __src, void * __dest, size_t __n) throw()
# 452
 __attribute((__nonnull__(1, 2))); 
# 455
extern void bzero(void * __s, size_t __n) throw() __attribute((__nonnull__(1))); 
# 458
extern int bcmp(const void * __s1, const void * __s2, size_t __n) throw()
# 459
 __attribute((__pure__)) __attribute((__nonnull__(1, 2))); 
# 463
extern "C++" {
# 465
extern char *index(char * __s, int __c) throw() __asm__("index")
# 466
 __attribute((__pure__)) __attribute((__nonnull__(1))); 
# 467
extern const char *index(const char * __s, int __c) throw() __asm__("index")
# 468
 __attribute((__pure__)) __attribute((__nonnull__(1))); 
# 483 "/usr/include/string.h" 3
}
# 491
extern "C++" {
# 493
extern char *rindex(char * __s, int __c) throw() __asm__("rindex")
# 494
 __attribute((__pure__)) __attribute((__nonnull__(1))); 
# 495
extern const char *rindex(const char * __s, int __c) throw() __asm__("rindex")
# 496
 __attribute((__pure__)) __attribute((__nonnull__(1))); 
# 511 "/usr/include/string.h" 3
}
# 519
extern int ffs(int __i) throw() __attribute((const)); 
# 524
extern int ffsl(long __l) throw() __attribute((const)); 
# 526
__extension__ extern int ffsll(long long __ll) throw()
# 527
 __attribute((const)); 
# 532
extern int strcasecmp(const char * __s1, const char * __s2) throw()
# 533
 __attribute((__pure__)) __attribute((__nonnull__(1, 2))); 
# 536
extern int strncasecmp(const char * __s1, const char * __s2, size_t __n) throw()
# 537
 __attribute((__pure__)) __attribute((__nonnull__(1, 2))); 
# 543
extern int strcasecmp_l(const char * __s1, const char * __s2, __locale_t __loc) throw()
# 545
 __attribute((__pure__)) __attribute((__nonnull__(1, 2, 3))); 
# 547
extern int strncasecmp_l(const char * __s1, const char * __s2, size_t __n, __locale_t __loc) throw()
# 549
 __attribute((__pure__)) __attribute((__nonnull__(1, 2, 4))); 
# 555
extern char *strsep(char **__restrict__ __stringp, const char *__restrict__ __delim) throw()
# 557
 __attribute((__nonnull__(1, 2))); 
# 562
extern char *strsignal(int __sig) throw(); 
# 565
extern char *__stpcpy(char *__restrict__ __dest, const char *__restrict__ __src) throw()
# 566
 __attribute((__nonnull__(1, 2))); 
# 567
extern char *stpcpy(char *__restrict__ __dest, const char *__restrict__ __src) throw()
# 568
 __attribute((__nonnull__(1, 2))); 
# 572
extern char *__stpncpy(char *__restrict__ __dest, const char *__restrict__ __src, size_t __n) throw()
# 574
 __attribute((__nonnull__(1, 2))); 
# 575
extern char *stpncpy(char *__restrict__ __dest, const char *__restrict__ __src, size_t __n) throw()
# 577
 __attribute((__nonnull__(1, 2))); 
# 582
extern int strverscmp(const char * __s1, const char * __s2) throw()
# 583
 __attribute((__pure__)) __attribute((__nonnull__(1, 2))); 
# 586
extern char *strfry(char * __string) throw() __attribute((__nonnull__(1))); 
# 589
extern void *memfrob(void * __s, size_t __n) throw() __attribute((__nonnull__(1))); 
# 597
extern "C++" char *basename(char * __filename) throw() __asm__("basename")
# 598
 __attribute((__nonnull__(1))); 
# 599
extern "C++" const char *basename(const char * __filename) throw() __asm__("basename")
# 600
 __attribute((__nonnull__(1))); 
# 642 "/usr/include/string.h" 3
}
# 29 "/usr/include/time.h" 3
extern "C" {
# 30 "/usr/include/bits/types.h" 3
typedef unsigned char __u_char; 
# 31
typedef unsigned short __u_short; 
# 32
typedef unsigned __u_int; 
# 33
typedef unsigned long __u_long; 
# 36
typedef signed char __int8_t; 
# 37
typedef unsigned char __uint8_t; 
# 38
typedef signed short __int16_t; 
# 39
typedef unsigned short __uint16_t; 
# 40
typedef signed int __int32_t; 
# 41
typedef unsigned __uint32_t; 
# 43
typedef signed long __int64_t; 
# 44
typedef unsigned long __uint64_t; 
# 52
typedef long __quad_t; 
# 53
typedef unsigned long __u_quad_t; 
# 133 "/usr/include/bits/types.h" 3
typedef unsigned long __dev_t; 
# 134
typedef unsigned __uid_t; 
# 135
typedef unsigned __gid_t; 
# 136
typedef unsigned long __ino_t; 
# 137
typedef unsigned long __ino64_t; 
# 138
typedef unsigned __mode_t; 
# 139
typedef unsigned long __nlink_t; 
# 140
typedef long __off_t; 
# 141
typedef long __off64_t; 
# 142
typedef int __pid_t; 
# 143
typedef struct { int __val[2]; } __fsid_t; 
# 144
typedef long __clock_t; 
# 145
typedef unsigned long __rlim_t; 
# 146
typedef unsigned long __rlim64_t; 
# 147
typedef unsigned __id_t; 
# 148
typedef long __time_t; 
# 149
typedef unsigned __useconds_t; 
# 150
typedef long __suseconds_t; 
# 152
typedef int __daddr_t; 
# 153
typedef int __key_t; 
# 156
typedef int __clockid_t; 
# 159
typedef void *__timer_t; 
# 162
typedef long __blksize_t; 
# 167
typedef long __blkcnt_t; 
# 168
typedef long __blkcnt64_t; 
# 171
typedef unsigned long __fsblkcnt_t; 
# 172
typedef unsigned long __fsblkcnt64_t; 
# 175
typedef unsigned long __fsfilcnt_t; 
# 176
typedef unsigned long __fsfilcnt64_t; 
# 179
typedef long __fsword_t; 
# 181
typedef long __ssize_t; 
# 184
typedef long __syscall_slong_t; 
# 186
typedef unsigned long __syscall_ulong_t; 
# 190
typedef __off64_t __loff_t; 
# 191
typedef __quad_t *__qaddr_t; 
# 192
typedef char *__caddr_t; 
# 195
typedef long __intptr_t; 
# 198
typedef unsigned __socklen_t; 
# 30 "/usr/include/bits/time.h" 3
struct timeval { 
# 32
__time_t tv_sec; 
# 33
__suseconds_t tv_usec; 
# 34
}; 
# 25 "/usr/include/bits/timex.h" 3
struct timex { 
# 27
unsigned modes; 
# 28
__syscall_slong_t offset; 
# 29
__syscall_slong_t freq; 
# 30
__syscall_slong_t maxerror; 
# 31
__syscall_slong_t esterror; 
# 32
int status; 
# 33
__syscall_slong_t constant; 
# 34
__syscall_slong_t precision; 
# 35
__syscall_slong_t tolerance; 
# 36
timeval time; 
# 37
__syscall_slong_t tick; 
# 38
__syscall_slong_t ppsfreq; 
# 39
__syscall_slong_t jitter; 
# 40
int shift; 
# 41
__syscall_slong_t stabil; 
# 42
__syscall_slong_t jitcnt; 
# 43
__syscall_slong_t calcnt; 
# 44
__syscall_slong_t errcnt; 
# 45
__syscall_slong_t stbcnt; 
# 47
int tai; 
# 50
int:32; int:32; int:32; int:32; 
# 51
int:32; int:32; int:32; int:32; 
# 52
int:32; int:32; int:32; 
# 53
}; 
# 90 "/usr/include/bits/time.h" 3
extern "C" {
# 93
extern int clock_adjtime(__clockid_t __clock_id, timex * __utx) throw(); 
# 95
}
# 59 "/usr/include/time.h" 3
typedef __clock_t clock_t; 
# 75 "/usr/include/time.h" 3
typedef __time_t time_t; 
# 91 "/usr/include/time.h" 3
typedef __clockid_t clockid_t; 
# 103 "/usr/include/time.h" 3
typedef __timer_t timer_t; 
# 120 "/usr/include/time.h" 3
struct timespec { 
# 122
__time_t tv_sec; 
# 123
__syscall_slong_t tv_nsec; 
# 124
}; 
# 133
struct tm { 
# 135
int tm_sec; 
# 136
int tm_min; 
# 137
int tm_hour; 
# 138
int tm_mday; 
# 139
int tm_mon; 
# 140
int tm_year; 
# 141
int tm_wday; 
# 142
int tm_yday; 
# 143
int tm_isdst; 
# 146
long tm_gmtoff; 
# 147
const char *tm_zone; 
# 152
}; 
# 161
struct itimerspec { 
# 163
timespec it_interval; 
# 164
timespec it_value; 
# 165
}; 
# 168
struct sigevent; 
# 174
typedef __pid_t pid_t; 
# 189 "/usr/include/time.h" 3
extern clock_t clock() throw(); 
# 192
extern time_t time(time_t * __timer) throw(); 
# 195
extern double difftime(time_t __time1, time_t __time0) throw()
# 196
 __attribute((const)); 
# 199
extern time_t mktime(tm * __tp) throw(); 
# 205
extern size_t strftime(char *__restrict__ __s, size_t __maxsize, const char *__restrict__ __format, const tm *__restrict__ __tp) throw(); 
# 213
extern char *strptime(const char *__restrict__ __s, const char *__restrict__ __fmt, tm * __tp) throw(); 
# 223
extern size_t strftime_l(char *__restrict__ __s, size_t __maxsize, const char *__restrict__ __format, const tm *__restrict__ __tp, __locale_t __loc) throw(); 
# 230
extern char *strptime_l(const char *__restrict__ __s, const char *__restrict__ __fmt, tm * __tp, __locale_t __loc) throw(); 
# 239
extern tm *gmtime(const time_t * __timer) throw(); 
# 243
extern tm *localtime(const time_t * __timer) throw(); 
# 249
extern tm *gmtime_r(const time_t *__restrict__ __timer, tm *__restrict__ __tp) throw(); 
# 254
extern tm *localtime_r(const time_t *__restrict__ __timer, tm *__restrict__ __tp) throw(); 
# 261
extern char *asctime(const tm * __tp) throw(); 
# 264
extern char *ctime(const time_t * __timer) throw(); 
# 272
extern char *asctime_r(const tm *__restrict__ __tp, char *__restrict__ __buf) throw(); 
# 276
extern char *ctime_r(const time_t *__restrict__ __timer, char *__restrict__ __buf) throw(); 
# 282
extern char *__tzname[2]; 
# 283
extern int __daylight; 
# 284
extern long __timezone; 
# 289
extern char *tzname[2]; 
# 293
extern void tzset() throw(); 
# 297
extern int daylight; 
# 298
extern long timezone; 
# 304
extern int stime(const time_t * __when) throw(); 
# 319 "/usr/include/time.h" 3
extern time_t timegm(tm * __tp) throw(); 
# 322
extern time_t timelocal(tm * __tp) throw(); 
# 325
extern int dysize(int __year) throw() __attribute((const)); 
# 334 "/usr/include/time.h" 3
extern int nanosleep(const timespec * __requested_time, timespec * __remaining); 
# 339
extern int clock_getres(clockid_t __clock_id, timespec * __res) throw(); 
# 342
extern int clock_gettime(clockid_t __clock_id, timespec * __tp) throw(); 
# 345
extern int clock_settime(clockid_t __clock_id, const timespec * __tp) throw(); 
# 353
extern int clock_nanosleep(clockid_t __clock_id, int __flags, const timespec * __req, timespec * __rem); 
# 358
extern int clock_getcpuclockid(pid_t __pid, clockid_t * __clock_id) throw(); 
# 363
extern int timer_create(clockid_t __clock_id, sigevent *__restrict__ __evp, timer_t *__restrict__ __timerid) throw(); 
# 368
extern int timer_delete(timer_t __timerid) throw(); 
# 371
extern int timer_settime(timer_t __timerid, int __flags, const itimerspec *__restrict__ __value, itimerspec *__restrict__ __ovalue) throw(); 
# 376
extern int timer_gettime(timer_t __timerid, itimerspec * __value) throw(); 
# 380
extern int timer_getoverrun(timer_t __timerid) throw(); 
# 386
extern int timespec_get(timespec * __ts, int __base) throw()
# 387
 __attribute((__nonnull__(1))); 
# 403 "/usr/include/time.h" 3
extern int getdate_err; 
# 412 "/usr/include/time.h" 3
extern tm *getdate(const char * __string); 
# 426 "/usr/include/time.h" 3
extern int getdate_r(const char *__restrict__ __string, tm *__restrict__ __resbufp); 
# 430
}
# 88 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/crt/common_functions.h"
extern "C" {
# 91
extern clock_t clock() throw(); 
# 96 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/crt/common_functions.h"
extern void *memset(void *, int, size_t) throw(); 
# 97 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/crt/common_functions.h"
extern void *memcpy(void *, const void *, size_t) throw(); 
# 99 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/crt/common_functions.h"
}
# 115 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/crt/math_functions.h"
extern "C" {
# 213 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/crt/math_functions.h"
extern int abs(int a) throw(); 
# 221 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/crt/math_functions.h"
extern long labs(long a) throw(); 
# 229 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/crt/math_functions.h"
extern long long llabs(long long a) throw(); 
# 279 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/crt/math_functions.h"
extern double fabs(double x) throw(); 
# 320 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/crt/math_functions.h"
extern float fabsf(float x) throw(); 
# 330 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/crt/math_functions.h"
extern inline int min(const int a, const int b); 
# 337
extern inline unsigned umin(const unsigned a, const unsigned b); 
# 344
extern inline long long llmin(const long long a, const long long b); 
# 351
extern inline unsigned long long ullmin(const unsigned long long a, const unsigned long long b); 
# 372 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/crt/math_functions.h"
extern float fminf(float x, float y) throw(); 
# 392 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/crt/math_functions.h"
extern double fmin(double x, double y) throw(); 
# 405 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/crt/math_functions.h"
extern inline int max(const int a, const int b); 
# 413
extern inline unsigned umax(const unsigned a, const unsigned b); 
# 420
extern inline long long llmax(const long long a, const long long b); 
# 427
extern inline unsigned long long ullmax(const unsigned long long a, const unsigned long long b); 
# 448 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/crt/math_functions.h"
extern float fmaxf(float x, float y) throw(); 
# 468 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/crt/math_functions.h"
extern double fmax(double, double) throw(); 
# 512 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/crt/math_functions.h"
extern double sin(double x) throw(); 
# 545 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/crt/math_functions.h"
extern double cos(double x) throw(); 
# 564 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/crt/math_functions.h"
extern void sincos(double x, double * sptr, double * cptr) throw(); 
# 580 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/crt/math_functions.h"
extern void sincosf(float x, float * sptr, float * cptr) throw(); 
# 625 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/crt/math_functions.h"
extern double tan(double x) throw(); 
# 694 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/crt/math_functions.h"
extern double sqrt(double x) throw(); 
# 766 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/crt/math_functions.h"
extern double rsqrt(double x); 
# 836 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/crt/math_functions.h"
extern float rsqrtf(float x); 
# 892 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/crt/math_functions.h"
extern double log2(double x) throw(); 
# 917 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/crt/math_functions.h"
extern double exp2(double x) throw(); 
# 942 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/crt/math_functions.h"
extern float exp2f(float x) throw(); 
# 969 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/crt/math_functions.h"
extern double exp10(double x) throw(); 
# 992 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/crt/math_functions.h"
extern float exp10f(float x) throw(); 
# 1038 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/crt/math_functions.h"
extern double expm1(double x) throw(); 
# 1083 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/crt/math_functions.h"
extern float expm1f(float x) throw(); 
# 1138 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/crt/math_functions.h"
extern float log2f(float x) throw(); 
# 1192 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/crt/math_functions.h"
extern double log10(double x) throw(); 
# 1263 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/crt/math_functions.h"
extern double log(double x) throw(); 
# 1366 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/crt/math_functions.h"
extern double log1p(double x) throw(); 
# 1472 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/crt/math_functions.h"
extern float log1pf(float x) throw(); 
# 1536 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/crt/math_functions.h"
extern double floor(double x) throw(); 
# 1575 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/crt/math_functions.h"
extern double exp(double x) throw(); 
# 1606 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/crt/math_functions.h"
extern double cosh(double x) throw(); 
# 1656 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/crt/math_functions.h"
extern double sinh(double x) throw(); 
# 1686 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/crt/math_functions.h"
extern double tanh(double x) throw(); 
# 1721 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/crt/math_functions.h"
extern double acosh(double x) throw(); 
# 1759 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/crt/math_functions.h"
extern float acoshf(float x) throw(); 
# 1775 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/crt/math_functions.h"
extern double asinh(double x) throw(); 
# 1791 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/crt/math_functions.h"
extern float asinhf(float x) throw(); 
# 1845 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/crt/math_functions.h"
extern double atanh(double x) throw(); 
# 1899 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/crt/math_functions.h"
extern float atanhf(float x) throw(); 
# 1958 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/crt/math_functions.h"
extern double ldexp(double x, int exp) throw(); 
# 2014 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/crt/math_functions.h"
extern float ldexpf(float x, int exp) throw(); 
# 2066 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/crt/math_functions.h"
extern double logb(double x) throw(); 
# 2121 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/crt/math_functions.h"
extern float logbf(float x) throw(); 
# 2152 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/crt/math_functions.h"
extern int ilogb(double x) throw(); 
# 2183 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/crt/math_functions.h"
extern int ilogbf(float x) throw(); 
# 2259 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/crt/math_functions.h"
extern double scalbn(double x, int n) throw(); 
# 2335 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/crt/math_functions.h"
extern float scalbnf(float x, int n) throw(); 
# 2411 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/crt/math_functions.h"
extern double scalbln(double x, long n) throw(); 
# 2487 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/crt/math_functions.h"
extern float scalblnf(float x, long n) throw(); 
# 2565 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/crt/math_functions.h"
extern double frexp(double x, int * nptr) throw(); 
# 2640 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/crt/math_functions.h"
extern float frexpf(float x, int * nptr) throw(); 
# 2654 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/crt/math_functions.h"
extern double round(double x) throw(); 
# 2671 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/crt/math_functions.h"
extern float roundf(float x) throw(); 
# 2689 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/crt/math_functions.h"
extern long lround(double x) throw(); 
# 2707 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/crt/math_functions.h"
extern long lroundf(float x) throw(); 
# 2725 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/crt/math_functions.h"
extern long long llround(double x) throw(); 
# 2743 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/crt/math_functions.h"
extern long long llroundf(float x) throw(); 
# 2795 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/crt/math_functions.h"
extern float rintf(float x) throw(); 
# 2812 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/crt/math_functions.h"
extern long lrint(double x) throw(); 
# 2829 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/crt/math_functions.h"
extern long lrintf(float x) throw(); 
# 2846 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/crt/math_functions.h"
extern long long llrint(double x) throw(); 
# 2863 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/crt/math_functions.h"
extern long long llrintf(float x) throw(); 
# 2916 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/crt/math_functions.h"
extern double nearbyint(double x) throw(); 
# 2969 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/crt/math_functions.h"
extern float nearbyintf(float x) throw(); 
# 3031 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/crt/math_functions.h"
extern double ceil(double x) throw(); 
# 3043 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/crt/math_functions.h"
extern double trunc(double x) throw(); 
# 3058 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/crt/math_functions.h"
extern float truncf(float x) throw(); 
# 3084 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/crt/math_functions.h"
extern double fdim(double x, double y) throw(); 
# 3110 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/crt/math_functions.h"
extern float fdimf(float x, float y) throw(); 
# 3146 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/crt/math_functions.h"
extern double atan2(double y, double x) throw(); 
# 3177 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/crt/math_functions.h"
extern double atan(double x) throw(); 
# 3200 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/crt/math_functions.h"
extern double acos(double x) throw(); 
# 3232 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/crt/math_functions.h"
extern double asin(double x) throw(); 
# 3278 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/crt/math_functions.h"
extern double hypot(double x, double y) throw(); 
# 3376 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/crt/math_functions.h"
extern float hypotf(float x, float y) throw(); 
# 4108 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/crt/math_functions.h"
extern double cbrt(double x) throw(); 
# 4194 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/crt/math_functions.h"
extern float cbrtf(float x) throw(); 
# 4249 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/crt/math_functions.h"
extern double rcbrt(double x); 
# 4299 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/crt/math_functions.h"
extern float rcbrtf(float x); 
# 4359 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/crt/math_functions.h"
extern double sinpi(double x); 
# 4419 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/crt/math_functions.h"
extern float sinpif(float x); 
# 4471 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/crt/math_functions.h"
extern double cospi(double x); 
# 4523 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/crt/math_functions.h"
extern float cospif(float x); 
# 4553 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/crt/math_functions.h"
extern void sincospi(double x, double * sptr, double * cptr); 
# 4583 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/crt/math_functions.h"
extern void sincospif(float x, float * sptr, float * cptr); 
# 4895 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/crt/math_functions.h"
extern double pow(double x, double y) throw(); 
# 4951 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/crt/math_functions.h"
extern double modf(double x, double * iptr) throw(); 
# 5010 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/crt/math_functions.h"
extern double fmod(double x, double y) throw(); 
# 5096 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/crt/math_functions.h"
extern double remainder(double x, double y) throw(); 
# 5186 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/crt/math_functions.h"
extern float remainderf(float x, float y) throw(); 
# 5240 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/crt/math_functions.h"
extern double remquo(double x, double y, int * quo) throw(); 
# 5294 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/crt/math_functions.h"
extern float remquof(float x, float y, int * quo) throw(); 
# 5335 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/crt/math_functions.h"
extern double j0(double x) throw(); 
# 5377 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/crt/math_functions.h"
extern float j0f(float x) throw(); 
# 5446 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/crt/math_functions.h"
extern double j1(double x) throw(); 
# 5515 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/crt/math_functions.h"
extern float j1f(float x) throw(); 
# 5558 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/crt/math_functions.h"
extern double jn(int n, double x) throw(); 
# 5601 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/crt/math_functions.h"
extern float jnf(int n, float x) throw(); 
# 5653 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/crt/math_functions.h"
extern double y0(double x) throw(); 
# 5705 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/crt/math_functions.h"
extern float y0f(float x) throw(); 
# 5757 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/crt/math_functions.h"
extern double y1(double x) throw(); 
# 5809 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/crt/math_functions.h"
extern float y1f(float x) throw(); 
# 5862 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/crt/math_functions.h"
extern double yn(int n, double x) throw(); 
# 5915 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/crt/math_functions.h"
extern float ynf(int n, float x) throw(); 
# 6104 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/crt/math_functions.h"
extern double erf(double x) throw(); 
# 6186 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/crt/math_functions.h"
extern float erff(float x) throw(); 
# 6250 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/crt/math_functions.h"
extern double erfinv(double y); 
# 6307 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/crt/math_functions.h"
extern float erfinvf(float y); 
# 6346 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/crt/math_functions.h"
extern double erfc(double x) throw(); 
# 6384 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/crt/math_functions.h"
extern float erfcf(float x) throw(); 
# 6512 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/crt/math_functions.h"
extern double lgamma(double x) throw(); 
# 6575 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/crt/math_functions.h"
extern double erfcinv(double y); 
# 6631 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/crt/math_functions.h"
extern float erfcinvf(float y); 
# 6689 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/crt/math_functions.h"
extern double normcdfinv(double y); 
# 6747 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/crt/math_functions.h"
extern float normcdfinvf(float y); 
# 6790 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/crt/math_functions.h"
extern double normcdf(double y); 
# 6833 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/crt/math_functions.h"
extern float normcdff(float y); 
# 6908 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/crt/math_functions.h"
extern double erfcx(double x); 
# 6983 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/crt/math_functions.h"
extern float erfcxf(float x); 
# 7117 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/crt/math_functions.h"
extern float lgammaf(float x) throw(); 
# 7226 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/crt/math_functions.h"
extern double tgamma(double x) throw(); 
# 7335 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/crt/math_functions.h"
extern float tgammaf(float x) throw(); 
# 7348 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/crt/math_functions.h"
extern double copysign(double x, double y) throw(); 
# 7361 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/crt/math_functions.h"
extern float copysignf(float x, float y) throw(); 
# 7380 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/crt/math_functions.h"
extern double nextafter(double x, double y) throw(); 
# 7399 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/crt/math_functions.h"
extern float nextafterf(float x, float y) throw(); 
# 7415 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/crt/math_functions.h"
extern double nan(const char * tagp) throw(); 
# 7431 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/crt/math_functions.h"
extern float nanf(const char * tagp) throw(); 
# 7438 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/crt/math_functions.h"
extern int __isinff(float) throw(); 
# 7439 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/crt/math_functions.h"
extern int __isnanf(float) throw(); 
# 7449 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/crt/math_functions.h"
extern int __finite(double) throw(); 
# 7450 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/crt/math_functions.h"
extern int __finitef(float) throw(); 
# 7451 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/crt/math_functions.h"
extern int __signbit(double) throw(); 
# 7452 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/crt/math_functions.h"
extern int __isnan(double) throw(); 
# 7453 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/crt/math_functions.h"
extern int __isinf(double) throw(); 
# 7456 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/crt/math_functions.h"
extern int __signbitf(float) throw(); 
# 7615 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/crt/math_functions.h"
extern double fma(double x, double y, double z) throw(); 
# 7773 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/crt/math_functions.h"
extern float fmaf(float x, float y, float z) throw(); 
# 7784 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/crt/math_functions.h"
extern int __signbitl(long double) throw(); 
# 7790 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/crt/math_functions.h"
extern int __finitel(long double) throw(); 
# 7791 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/crt/math_functions.h"
extern int __isinfl(long double) throw(); 
# 7792 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/crt/math_functions.h"
extern int __isnanl(long double) throw(); 
# 7842 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/crt/math_functions.h"
extern float acosf(float x) throw(); 
# 7882 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/crt/math_functions.h"
extern float asinf(float x) throw(); 
# 7922 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/crt/math_functions.h"
extern float atanf(float x) throw(); 
# 7955 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/crt/math_functions.h"
extern float atan2f(float y, float x) throw(); 
# 7979 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/crt/math_functions.h"
extern float cosf(float x) throw(); 
# 8021 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/crt/math_functions.h"
extern float sinf(float x) throw(); 
# 8063 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/crt/math_functions.h"
extern float tanf(float x) throw(); 
# 8094 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/crt/math_functions.h"
extern float coshf(float x) throw(); 
# 8144 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/crt/math_functions.h"
extern float sinhf(float x) throw(); 
# 8174 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/crt/math_functions.h"
extern float tanhf(float x) throw(); 
# 8225 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/crt/math_functions.h"
extern float logf(float x) throw(); 
# 8275 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/crt/math_functions.h"
extern float expf(float x) throw(); 
# 8326 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/crt/math_functions.h"
extern float log10f(float x) throw(); 
# 8381 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/crt/math_functions.h"
extern float modff(float x, float * iptr) throw(); 
# 8689 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/crt/math_functions.h"
extern float powf(float x, float y) throw(); 
# 8758 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/crt/math_functions.h"
extern float sqrtf(float x) throw(); 
# 8817 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/crt/math_functions.h"
extern float ceilf(float x) throw(); 
# 8878 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/crt/math_functions.h"
extern float floorf(float x) throw(); 
# 8936 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/crt/math_functions.h"
extern float fmodf(float x, float y) throw(); 
# 8951 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/crt/math_functions.h"
}
# 2190 "/opt/rh/devtoolset-7/root/usr/include/c++/7/x86_64-redhat-linux/bits/c++config.h" 3
namespace std { 
# 2192
typedef unsigned long size_t; 
# 2193
typedef long ptrdiff_t; 
# 2196
typedef __decltype((nullptr)) nullptr_t; 
# 2198
}
# 67 "/opt/rh/devtoolset-7/root/usr/include/c++/7/bits/cpp_type_traits.h" 3
extern "C++" {
# 69
namespace std __attribute((__visibility__("default"))) { 
# 73
struct __true_type { }; 
# 74
struct __false_type { }; 
# 76
template< bool > 
# 77
struct __truth_type { 
# 78
typedef __false_type __type; }; 
# 81
template<> struct __truth_type< true>  { 
# 82
typedef __true_type __type; }; 
# 86
template< class _Sp, class _Tp> 
# 87
struct __traitor { 
# 89
enum { __value = ((bool)_Sp::__value) || ((bool)_Tp::__value)}; 
# 90
typedef typename __truth_type< __value> ::__type __type; 
# 91
}; 
# 94
template< class , class > 
# 95
struct __are_same { 
# 97
enum { __value}; 
# 98
typedef __false_type __type; 
# 99
}; 
# 101
template< class _Tp> 
# 102
struct __are_same< _Tp, _Tp>  { 
# 104
enum { __value = 1}; 
# 105
typedef __true_type __type; 
# 106
}; 
# 109
template< class _Tp> 
# 110
struct __is_void { 
# 112
enum { __value}; 
# 113
typedef __false_type __type; 
# 114
}; 
# 117
template<> struct __is_void< void>  { 
# 119
enum { __value = 1}; 
# 120
typedef __true_type __type; 
# 121
}; 
# 126
template< class _Tp> 
# 127
struct __is_integer { 
# 129
enum { __value}; 
# 130
typedef __false_type __type; 
# 131
}; 
# 138
template<> struct __is_integer< bool>  { 
# 140
enum { __value = 1}; 
# 141
typedef __true_type __type; 
# 142
}; 
# 145
template<> struct __is_integer< char>  { 
# 147
enum { __value = 1}; 
# 148
typedef __true_type __type; 
# 149
}; 
# 152
template<> struct __is_integer< signed char>  { 
# 154
enum { __value = 1}; 
# 155
typedef __true_type __type; 
# 156
}; 
# 159
template<> struct __is_integer< unsigned char>  { 
# 161
enum { __value = 1}; 
# 162
typedef __true_type __type; 
# 163
}; 
# 167
template<> struct __is_integer< wchar_t>  { 
# 169
enum { __value = 1}; 
# 170
typedef __true_type __type; 
# 171
}; 
# 176
template<> struct __is_integer< char16_t>  { 
# 178
enum { __value = 1}; 
# 179
typedef __true_type __type; 
# 180
}; 
# 183
template<> struct __is_integer< char32_t>  { 
# 185
enum { __value = 1}; 
# 186
typedef __true_type __type; 
# 187
}; 
# 191
template<> struct __is_integer< short>  { 
# 193
enum { __value = 1}; 
# 194
typedef __true_type __type; 
# 195
}; 
# 198
template<> struct __is_integer< unsigned short>  { 
# 200
enum { __value = 1}; 
# 201
typedef __true_type __type; 
# 202
}; 
# 205
template<> struct __is_integer< int>  { 
# 207
enum { __value = 1}; 
# 208
typedef __true_type __type; 
# 209
}; 
# 212
template<> struct __is_integer< unsigned>  { 
# 214
enum { __value = 1}; 
# 215
typedef __true_type __type; 
# 216
}; 
# 219
template<> struct __is_integer< long>  { 
# 221
enum { __value = 1}; 
# 222
typedef __true_type __type; 
# 223
}; 
# 226
template<> struct __is_integer< unsigned long>  { 
# 228
enum { __value = 1}; 
# 229
typedef __true_type __type; 
# 230
}; 
# 233
template<> struct __is_integer< long long>  { 
# 235
enum { __value = 1}; 
# 236
typedef __true_type __type; 
# 237
}; 
# 240
template<> struct __is_integer< unsigned long long>  { 
# 242
enum { __value = 1}; 
# 243
typedef __true_type __type; 
# 244
}; 
# 261 "/opt/rh/devtoolset-7/root/usr/include/c++/7/bits/cpp_type_traits.h" 3
template<> struct __is_integer< __int128>  { enum { __value = 1}; typedef __true_type __type; }; template<> struct __is_integer< unsigned __int128>  { enum { __value = 1}; typedef __true_type __type; }; 
# 278 "/opt/rh/devtoolset-7/root/usr/include/c++/7/bits/cpp_type_traits.h" 3
template< class _Tp> 
# 279
struct __is_floating { 
# 281
enum { __value}; 
# 282
typedef __false_type __type; 
# 283
}; 
# 287
template<> struct __is_floating< float>  { 
# 289
enum { __value = 1}; 
# 290
typedef __true_type __type; 
# 291
}; 
# 294
template<> struct __is_floating< double>  { 
# 296
enum { __value = 1}; 
# 297
typedef __true_type __type; 
# 298
}; 
# 301
template<> struct __is_floating< long double>  { 
# 303
enum { __value = 1}; 
# 304
typedef __true_type __type; 
# 305
}; 
# 310
template< class _Tp> 
# 311
struct __is_pointer { 
# 313
enum { __value}; 
# 314
typedef __false_type __type; 
# 315
}; 
# 317
template< class _Tp> 
# 318
struct __is_pointer< _Tp *>  { 
# 320
enum { __value = 1}; 
# 321
typedef __true_type __type; 
# 322
}; 
# 327
template< class _Tp> 
# 328
struct __is_arithmetic : public __traitor< __is_integer< _Tp> , __is_floating< _Tp> >  { 
# 330
}; 
# 335
template< class _Tp> 
# 336
struct __is_scalar : public __traitor< __is_arithmetic< _Tp> , __is_pointer< _Tp> >  { 
# 338
}; 
# 343
template< class _Tp> 
# 344
struct __is_char { 
# 346
enum { __value}; 
# 347
typedef __false_type __type; 
# 348
}; 
# 351
template<> struct __is_char< char>  { 
# 353
enum { __value = 1}; 
# 354
typedef __true_type __type; 
# 355
}; 
# 359
template<> struct __is_char< wchar_t>  { 
# 361
enum { __value = 1}; 
# 362
typedef __true_type __type; 
# 363
}; 
# 366
template< class _Tp> 
# 367
struct __is_byte { 
# 369
enum { __value}; 
# 370
typedef __false_type __type; 
# 371
}; 
# 374
template<> struct __is_byte< char>  { 
# 376
enum { __value = 1}; 
# 377
typedef __true_type __type; 
# 378
}; 
# 381
template<> struct __is_byte< signed char>  { 
# 383
enum { __value = 1}; 
# 384
typedef __true_type __type; 
# 385
}; 
# 388
template<> struct __is_byte< unsigned char>  { 
# 390
enum { __value = 1}; 
# 391
typedef __true_type __type; 
# 392
}; 
# 397
template< class _Tp> 
# 398
struct __is_move_iterator { 
# 400
enum { __value}; 
# 401
typedef __false_type __type; 
# 402
}; 
# 406
template< class _Iterator> inline _Iterator 
# 408
__miter_base(_Iterator __it) 
# 409
{ return __it; } 
# 412
}
# 413
}
# 37 "/opt/rh/devtoolset-7/root/usr/include/c++/7/ext/type_traits.h" 3
extern "C++" {
# 39
namespace __gnu_cxx __attribute((__visibility__("default"))) { 
# 44
template< bool , class > 
# 45
struct __enable_if { 
# 46
}; 
# 48
template< class _Tp> 
# 49
struct __enable_if< true, _Tp>  { 
# 50
typedef _Tp __type; }; 
# 54
template< bool _Cond, class _Iftrue, class _Iffalse> 
# 55
struct __conditional_type { 
# 56
typedef _Iftrue __type; }; 
# 58
template< class _Iftrue, class _Iffalse> 
# 59
struct __conditional_type< false, _Iftrue, _Iffalse>  { 
# 60
typedef _Iffalse __type; }; 
# 64
template< class _Tp> 
# 65
struct __add_unsigned { 
# 68
private: typedef __enable_if< std::__is_integer< _Tp> ::__value, _Tp>  __if_type; 
# 71
public: typedef typename __enable_if< std::__is_integer< _Tp> ::__value, _Tp> ::__type __type; 
# 72
}; 
# 75
template<> struct __add_unsigned< char>  { 
# 76
typedef unsigned char __type; }; 
# 79
template<> struct __add_unsigned< signed char>  { 
# 80
typedef unsigned char __type; }; 
# 83
template<> struct __add_unsigned< short>  { 
# 84
typedef unsigned short __type; }; 
# 87
template<> struct __add_unsigned< int>  { 
# 88
typedef unsigned __type; }; 
# 91
template<> struct __add_unsigned< long>  { 
# 92
typedef unsigned long __type; }; 
# 95
template<> struct __add_unsigned< long long>  { 
# 96
typedef unsigned long long __type; }; 
# 100
template<> struct __add_unsigned< bool> ; 
# 103
template<> struct __add_unsigned< wchar_t> ; 
# 107
template< class _Tp> 
# 108
struct __remove_unsigned { 
# 111
private: typedef __enable_if< std::__is_integer< _Tp> ::__value, _Tp>  __if_type; 
# 114
public: typedef typename __enable_if< std::__is_integer< _Tp> ::__value, _Tp> ::__type __type; 
# 115
}; 
# 118
template<> struct __remove_unsigned< char>  { 
# 119
typedef signed char __type; }; 
# 122
template<> struct __remove_unsigned< unsigned char>  { 
# 123
typedef signed char __type; }; 
# 126
template<> struct __remove_unsigned< unsigned short>  { 
# 127
typedef short __type; }; 
# 130
template<> struct __remove_unsigned< unsigned>  { 
# 131
typedef int __type; }; 
# 134
template<> struct __remove_unsigned< unsigned long>  { 
# 135
typedef long __type; }; 
# 138
template<> struct __remove_unsigned< unsigned long long>  { 
# 139
typedef long long __type; }; 
# 143
template<> struct __remove_unsigned< bool> ; 
# 146
template<> struct __remove_unsigned< wchar_t> ; 
# 150
template< class _Type> inline bool 
# 152
__is_null_pointer(_Type *__ptr) 
# 153
{ return __ptr == 0; } 
# 155
template< class _Type> inline bool 
# 157
__is_null_pointer(_Type) 
# 158
{ return false; } 
# 162
inline bool __is_null_pointer(std::nullptr_t) 
# 163
{ return true; } 
# 167
template< class _Tp, bool  = std::__is_integer< _Tp> ::__value> 
# 168
struct __promote { 
# 169
typedef double __type; }; 
# 174
template< class _Tp> 
# 175
struct __promote< _Tp, false>  { 
# 176
}; 
# 179
template<> struct __promote< long double>  { 
# 180
typedef long double __type; }; 
# 183
template<> struct __promote< double>  { 
# 184
typedef double __type; }; 
# 187
template<> struct __promote< float>  { 
# 188
typedef float __type; }; 
# 190
template< class _Tp, class _Up, class 
# 191
_Tp2 = typename __promote< _Tp> ::__type, class 
# 192
_Up2 = typename __promote< _Up> ::__type> 
# 193
struct __promote_2 { 
# 195
typedef __typeof__(_Tp2() + _Up2()) __type; 
# 196
}; 
# 198
template< class _Tp, class _Up, class _Vp, class 
# 199
_Tp2 = typename __promote< _Tp> ::__type, class 
# 200
_Up2 = typename __promote< _Up> ::__type, class 
# 201
_Vp2 = typename __promote< _Vp> ::__type> 
# 202
struct __promote_3 { 
# 204
typedef __typeof__((_Tp2() + _Up2()) + _Vp2()) __type; 
# 205
}; 
# 207
template< class _Tp, class _Up, class _Vp, class _Wp, class 
# 208
_Tp2 = typename __promote< _Tp> ::__type, class 
# 209
_Up2 = typename __promote< _Up> ::__type, class 
# 210
_Vp2 = typename __promote< _Vp> ::__type, class 
# 211
_Wp2 = typename __promote< _Wp> ::__type> 
# 212
struct __promote_4 { 
# 214
typedef __typeof__(((_Tp2() + _Up2()) + _Vp2()) + _Wp2()) __type; 
# 215
}; 
# 218
}
# 219
}
# 29 "/usr/include/math.h" 3
extern "C" {
# 28 "/usr/include/bits/mathdef.h" 3
typedef float float_t; 
# 29
typedef double double_t; 
# 54 "/usr/include/bits/mathcalls.h" 3
extern double acos(double __x) throw(); extern double __acos(double __x) throw(); 
# 56
extern double asin(double __x) throw(); extern double __asin(double __x) throw(); 
# 58
extern double atan(double __x) throw(); extern double __atan(double __x) throw(); 
# 60
extern double atan2(double __y, double __x) throw(); extern double __atan2(double __y, double __x) throw(); 
# 63
extern double cos(double __x) throw(); extern double __cos(double __x) throw(); 
# 65
extern double sin(double __x) throw(); extern double __sin(double __x) throw(); 
# 67
extern double tan(double __x) throw(); extern double __tan(double __x) throw(); 
# 72
extern double cosh(double __x) throw(); extern double __cosh(double __x) throw(); 
# 74
extern double sinh(double __x) throw(); extern double __sinh(double __x) throw(); 
# 76
extern double tanh(double __x) throw(); extern double __tanh(double __x) throw(); 
# 81
extern void sincos(double __x, double * __sinx, double * __cosx) throw(); extern void __sincos(double __x, double * __sinx, double * __cosx) throw(); 
# 88
extern double acosh(double __x) throw(); extern double __acosh(double __x) throw(); 
# 90
extern double asinh(double __x) throw(); extern double __asinh(double __x) throw(); 
# 92
extern double atanh(double __x) throw(); extern double __atanh(double __x) throw(); 
# 100
extern double exp(double __x) throw(); extern double __exp(double __x) throw(); 
# 103
extern double frexp(double __x, int * __exponent) throw(); extern double __frexp(double __x, int * __exponent) throw(); 
# 106
extern double ldexp(double __x, int __exponent) throw(); extern double __ldexp(double __x, int __exponent) throw(); 
# 109
extern double log(double __x) throw(); extern double __log(double __x) throw(); 
# 112
extern double log10(double __x) throw(); extern double __log10(double __x) throw(); 
# 115
extern double modf(double __x, double * __iptr) throw(); extern double __modf(double __x, double * __iptr) throw()
# 116
 __attribute((__nonnull__(2))); 
# 121
extern double exp10(double __x) throw(); extern double __exp10(double __x) throw(); 
# 123
extern double pow10(double __x) throw(); extern double __pow10(double __x) throw(); 
# 129
extern double expm1(double __x) throw(); extern double __expm1(double __x) throw(); 
# 132
extern double log1p(double __x) throw(); extern double __log1p(double __x) throw(); 
# 135
extern double logb(double __x) throw(); extern double __logb(double __x) throw(); 
# 142
extern double exp2(double __x) throw(); extern double __exp2(double __x) throw(); 
# 145
extern double log2(double __x) throw(); extern double __log2(double __x) throw(); 
# 154
extern double pow(double __x, double __y) throw(); extern double __pow(double __x, double __y) throw(); 
# 157
extern double sqrt(double __x) throw(); extern double __sqrt(double __x) throw(); 
# 163
extern double hypot(double __x, double __y) throw(); extern double __hypot(double __x, double __y) throw(); 
# 170
extern double cbrt(double __x) throw(); extern double __cbrt(double __x) throw(); 
# 179
extern double ceil(double __x) throw() __attribute((const)); extern double __ceil(double __x) throw() __attribute((const)); 
# 182
extern double fabs(double __x) throw() __attribute((const)); extern double __fabs(double __x) throw() __attribute((const)); 
# 185
extern double floor(double __x) throw() __attribute((const)); extern double __floor(double __x) throw() __attribute((const)); 
# 188
extern double fmod(double __x, double __y) throw(); extern double __fmod(double __x, double __y) throw(); 
# 193
extern int __isinf(double __value) throw() __attribute((const)); 
# 196
extern int __finite(double __value) throw() __attribute((const)); 
# 202
extern int isinf(double __value) throw() __attribute((const)); 
# 205
extern int finite(double __value) throw() __attribute((const)); 
# 208
extern double drem(double __x, double __y) throw(); extern double __drem(double __x, double __y) throw(); 
# 212
extern double significand(double __x) throw(); extern double __significand(double __x) throw(); 
# 218
extern double copysign(double __x, double __y) throw() __attribute((const)); extern double __copysign(double __x, double __y) throw() __attribute((const)); 
# 225
extern double nan(const char * __tagb) throw() __attribute((const)); extern double __nan(const char * __tagb) throw() __attribute((const)); 
# 231
extern int __isnan(double __value) throw() __attribute((const)); 
# 235
extern int isnan(double __value) throw() __attribute((const)); 
# 238
extern double j0(double) throw(); extern double __j0(double) throw(); 
# 239
extern double j1(double) throw(); extern double __j1(double) throw(); 
# 240
extern double jn(int, double) throw(); extern double __jn(int, double) throw(); 
# 241
extern double y0(double) throw(); extern double __y0(double) throw(); 
# 242
extern double y1(double) throw(); extern double __y1(double) throw(); 
# 243
extern double yn(int, double) throw(); extern double __yn(int, double) throw(); 
# 250
extern double erf(double) throw(); extern double __erf(double) throw(); 
# 251
extern double erfc(double) throw(); extern double __erfc(double) throw(); 
# 252
extern double lgamma(double) throw(); extern double __lgamma(double) throw(); 
# 259
extern double tgamma(double) throw(); extern double __tgamma(double) throw(); 
# 265
extern double gamma(double) throw(); extern double __gamma(double) throw(); 
# 272
extern double lgamma_r(double, int * __signgamp) throw(); extern double __lgamma_r(double, int * __signgamp) throw(); 
# 280
extern double rint(double __x) throw(); extern double __rint(double __x) throw(); 
# 283
extern double nextafter(double __x, double __y) throw() __attribute((const)); extern double __nextafter(double __x, double __y) throw() __attribute((const)); 
# 285
extern double nexttoward(double __x, long double __y) throw() __attribute((const)); extern double __nexttoward(double __x, long double __y) throw() __attribute((const)); 
# 289
extern double remainder(double __x, double __y) throw(); extern double __remainder(double __x, double __y) throw(); 
# 293
extern double scalbn(double __x, int __n) throw(); extern double __scalbn(double __x, int __n) throw(); 
# 297
extern int ilogb(double __x) throw(); extern int __ilogb(double __x) throw(); 
# 302
extern double scalbln(double __x, long __n) throw(); extern double __scalbln(double __x, long __n) throw(); 
# 306
extern double nearbyint(double __x) throw(); extern double __nearbyint(double __x) throw(); 
# 310
extern double round(double __x) throw() __attribute((const)); extern double __round(double __x) throw() __attribute((const)); 
# 314
extern double trunc(double __x) throw() __attribute((const)); extern double __trunc(double __x) throw() __attribute((const)); 
# 319
extern double remquo(double __x, double __y, int * __quo) throw(); extern double __remquo(double __x, double __y, int * __quo) throw(); 
# 326
extern long lrint(double __x) throw(); extern long __lrint(double __x) throw(); 
# 327
extern long long llrint(double __x) throw(); extern long long __llrint(double __x) throw(); 
# 331
extern long lround(double __x) throw(); extern long __lround(double __x) throw(); 
# 332
extern long long llround(double __x) throw(); extern long long __llround(double __x) throw(); 
# 336
extern double fdim(double __x, double __y) throw(); extern double __fdim(double __x, double __y) throw(); 
# 339
extern double fmax(double __x, double __y) throw() __attribute((const)); extern double __fmax(double __x, double __y) throw() __attribute((const)); 
# 342
extern double fmin(double __x, double __y) throw() __attribute((const)); extern double __fmin(double __x, double __y) throw() __attribute((const)); 
# 346
extern int __fpclassify(double __value) throw()
# 347
 __attribute((const)); 
# 350
extern int __signbit(double __value) throw()
# 351
 __attribute((const)); 
# 355
extern double fma(double __x, double __y, double __z) throw(); extern double __fma(double __x, double __y, double __z) throw(); 
# 364
extern double scalb(double __x, double __n) throw(); extern double __scalb(double __x, double __n) throw(); 
# 54 "/usr/include/bits/mathcalls.h" 3
extern float acosf(float __x) throw(); extern float __acosf(float __x) throw(); 
# 56
extern float asinf(float __x) throw(); extern float __asinf(float __x) throw(); 
# 58
extern float atanf(float __x) throw(); extern float __atanf(float __x) throw(); 
# 60
extern float atan2f(float __y, float __x) throw(); extern float __atan2f(float __y, float __x) throw(); 
# 63
extern float cosf(float __x) throw(); 
# 65
extern float sinf(float __x) throw(); 
# 67
extern float tanf(float __x) throw(); 
# 72
extern float coshf(float __x) throw(); extern float __coshf(float __x) throw(); 
# 74
extern float sinhf(float __x) throw(); extern float __sinhf(float __x) throw(); 
# 76
extern float tanhf(float __x) throw(); extern float __tanhf(float __x) throw(); 
# 81
extern void sincosf(float __x, float * __sinx, float * __cosx) throw(); 
# 88
extern float acoshf(float __x) throw(); extern float __acoshf(float __x) throw(); 
# 90
extern float asinhf(float __x) throw(); extern float __asinhf(float __x) throw(); 
# 92
extern float atanhf(float __x) throw(); extern float __atanhf(float __x) throw(); 
# 100
extern float expf(float __x) throw(); 
# 103
extern float frexpf(float __x, int * __exponent) throw(); extern float __frexpf(float __x, int * __exponent) throw(); 
# 106
extern float ldexpf(float __x, int __exponent) throw(); extern float __ldexpf(float __x, int __exponent) throw(); 
# 109
extern float logf(float __x) throw(); 
# 112
extern float log10f(float __x) throw(); 
# 115
extern float modff(float __x, float * __iptr) throw(); extern float __modff(float __x, float * __iptr) throw()
# 116
 __attribute((__nonnull__(2))); 
# 121
extern float exp10f(float __x) throw(); 
# 123
extern float pow10f(float __x) throw(); extern float __pow10f(float __x) throw(); 
# 129
extern float expm1f(float __x) throw(); extern float __expm1f(float __x) throw(); 
# 132
extern float log1pf(float __x) throw(); extern float __log1pf(float __x) throw(); 
# 135
extern float logbf(float __x) throw(); extern float __logbf(float __x) throw(); 
# 142
extern float exp2f(float __x) throw(); extern float __exp2f(float __x) throw(); 
# 145
extern float log2f(float __x) throw(); 
# 154
extern float powf(float __x, float __y) throw(); 
# 157
extern float sqrtf(float __x) throw(); extern float __sqrtf(float __x) throw(); 
# 163
extern float hypotf(float __x, float __y) throw(); extern float __hypotf(float __x, float __y) throw(); 
# 170
extern float cbrtf(float __x) throw(); extern float __cbrtf(float __x) throw(); 
# 179
extern float ceilf(float __x) throw() __attribute((const)); extern float __ceilf(float __x) throw() __attribute((const)); 
# 182
extern float fabsf(float __x) throw() __attribute((const)); extern float __fabsf(float __x) throw() __attribute((const)); 
# 185
extern float floorf(float __x) throw() __attribute((const)); extern float __floorf(float __x) throw() __attribute((const)); 
# 188
extern float fmodf(float __x, float __y) throw(); extern float __fmodf(float __x, float __y) throw(); 
# 193
extern int __isinff(float __value) throw() __attribute((const)); 
# 196
extern int __finitef(float __value) throw() __attribute((const)); 
# 202
extern int isinff(float __value) throw() __attribute((const)); 
# 205
extern int finitef(float __value) throw() __attribute((const)); 
# 208
extern float dremf(float __x, float __y) throw(); extern float __dremf(float __x, float __y) throw(); 
# 212
extern float significandf(float __x) throw(); extern float __significandf(float __x) throw(); 
# 218
extern float copysignf(float __x, float __y) throw() __attribute((const)); extern float __copysignf(float __x, float __y) throw() __attribute((const)); 
# 225
extern float nanf(const char * __tagb) throw() __attribute((const)); extern float __nanf(const char * __tagb) throw() __attribute((const)); 
# 231
extern int __isnanf(float __value) throw() __attribute((const)); 
# 235
extern int isnanf(float __value) throw() __attribute((const)); 
# 238
extern float j0f(float) throw(); extern float __j0f(float) throw(); 
# 239
extern float j1f(float) throw(); extern float __j1f(float) throw(); 
# 240
extern float jnf(int, float) throw(); extern float __jnf(int, float) throw(); 
# 241
extern float y0f(float) throw(); extern float __y0f(float) throw(); 
# 242
extern float y1f(float) throw(); extern float __y1f(float) throw(); 
# 243
extern float ynf(int, float) throw(); extern float __ynf(int, float) throw(); 
# 250
extern float erff(float) throw(); extern float __erff(float) throw(); 
# 251
extern float erfcf(float) throw(); extern float __erfcf(float) throw(); 
# 252
extern float lgammaf(float) throw(); extern float __lgammaf(float) throw(); 
# 259
extern float tgammaf(float) throw(); extern float __tgammaf(float) throw(); 
# 265
extern float gammaf(float) throw(); extern float __gammaf(float) throw(); 
# 272
extern float lgammaf_r(float, int * __signgamp) throw(); extern float __lgammaf_r(float, int * __signgamp) throw(); 
# 280
extern float rintf(float __x) throw(); extern float __rintf(float __x) throw(); 
# 283
extern float nextafterf(float __x, float __y) throw() __attribute((const)); extern float __nextafterf(float __x, float __y) throw() __attribute((const)); 
# 285
extern float nexttowardf(float __x, long double __y) throw() __attribute((const)); extern float __nexttowardf(float __x, long double __y) throw() __attribute((const)); 
# 289
extern float remainderf(float __x, float __y) throw(); extern float __remainderf(float __x, float __y) throw(); 
# 293
extern float scalbnf(float __x, int __n) throw(); extern float __scalbnf(float __x, int __n) throw(); 
# 297
extern int ilogbf(float __x) throw(); extern int __ilogbf(float __x) throw(); 
# 302
extern float scalblnf(float __x, long __n) throw(); extern float __scalblnf(float __x, long __n) throw(); 
# 306
extern float nearbyintf(float __x) throw(); extern float __nearbyintf(float __x) throw(); 
# 310
extern float roundf(float __x) throw() __attribute((const)); extern float __roundf(float __x) throw() __attribute((const)); 
# 314
extern float truncf(float __x) throw() __attribute((const)); extern float __truncf(float __x) throw() __attribute((const)); 
# 319
extern float remquof(float __x, float __y, int * __quo) throw(); extern float __remquof(float __x, float __y, int * __quo) throw(); 
# 326
extern long lrintf(float __x) throw(); extern long __lrintf(float __x) throw(); 
# 327
extern long long llrintf(float __x) throw(); extern long long __llrintf(float __x) throw(); 
# 331
extern long lroundf(float __x) throw(); extern long __lroundf(float __x) throw(); 
# 332
extern long long llroundf(float __x) throw(); extern long long __llroundf(float __x) throw(); 
# 336
extern float fdimf(float __x, float __y) throw(); extern float __fdimf(float __x, float __y) throw(); 
# 339
extern float fmaxf(float __x, float __y) throw() __attribute((const)); extern float __fmaxf(float __x, float __y) throw() __attribute((const)); 
# 342
extern float fminf(float __x, float __y) throw() __attribute((const)); extern float __fminf(float __x, float __y) throw() __attribute((const)); 
# 346
extern int __fpclassifyf(float __value) throw()
# 347
 __attribute((const)); 
# 350
extern int __signbitf(float __value) throw()
# 351
 __attribute((const)); 
# 355
extern float fmaf(float __x, float __y, float __z) throw(); extern float __fmaf(float __x, float __y, float __z) throw(); 
# 364
extern float scalbf(float __x, float __n) throw(); extern float __scalbf(float __x, float __n) throw(); 
# 54 "/usr/include/bits/mathcalls.h" 3
extern long double acosl(long double __x) throw(); extern long double __acosl(long double __x) throw(); 
# 56
extern long double asinl(long double __x) throw(); extern long double __asinl(long double __x) throw(); 
# 58
extern long double atanl(long double __x) throw(); extern long double __atanl(long double __x) throw(); 
# 60
extern long double atan2l(long double __y, long double __x) throw(); extern long double __atan2l(long double __y, long double __x) throw(); 
# 63
extern long double cosl(long double __x) throw(); extern long double __cosl(long double __x) throw(); 
# 65
extern long double sinl(long double __x) throw(); extern long double __sinl(long double __x) throw(); 
# 67
extern long double tanl(long double __x) throw(); extern long double __tanl(long double __x) throw(); 
# 72
extern long double coshl(long double __x) throw(); extern long double __coshl(long double __x) throw(); 
# 74
extern long double sinhl(long double __x) throw(); extern long double __sinhl(long double __x) throw(); 
# 76
extern long double tanhl(long double __x) throw(); extern long double __tanhl(long double __x) throw(); 
# 81
extern void sincosl(long double __x, long double * __sinx, long double * __cosx) throw(); extern void __sincosl(long double __x, long double * __sinx, long double * __cosx) throw(); 
# 88
extern long double acoshl(long double __x) throw(); extern long double __acoshl(long double __x) throw(); 
# 90
extern long double asinhl(long double __x) throw(); extern long double __asinhl(long double __x) throw(); 
# 92
extern long double atanhl(long double __x) throw(); extern long double __atanhl(long double __x) throw(); 
# 100
extern long double expl(long double __x) throw(); extern long double __expl(long double __x) throw(); 
# 103
extern long double frexpl(long double __x, int * __exponent) throw(); extern long double __frexpl(long double __x, int * __exponent) throw(); 
# 106
extern long double ldexpl(long double __x, int __exponent) throw(); extern long double __ldexpl(long double __x, int __exponent) throw(); 
# 109
extern long double logl(long double __x) throw(); extern long double __logl(long double __x) throw(); 
# 112
extern long double log10l(long double __x) throw(); extern long double __log10l(long double __x) throw(); 
# 115
extern long double modfl(long double __x, long double * __iptr) throw(); extern long double __modfl(long double __x, long double * __iptr) throw()
# 116
 __attribute((__nonnull__(2))); 
# 121
extern long double exp10l(long double __x) throw(); extern long double __exp10l(long double __x) throw(); 
# 123
extern long double pow10l(long double __x) throw(); extern long double __pow10l(long double __x) throw(); 
# 129
extern long double expm1l(long double __x) throw(); extern long double __expm1l(long double __x) throw(); 
# 132
extern long double log1pl(long double __x) throw(); extern long double __log1pl(long double __x) throw(); 
# 135
extern long double logbl(long double __x) throw(); extern long double __logbl(long double __x) throw(); 
# 142
extern long double exp2l(long double __x) throw(); extern long double __exp2l(long double __x) throw(); 
# 145
extern long double log2l(long double __x) throw(); extern long double __log2l(long double __x) throw(); 
# 154
extern long double powl(long double __x, long double __y) throw(); extern long double __powl(long double __x, long double __y) throw(); 
# 157
extern long double sqrtl(long double __x) throw(); extern long double __sqrtl(long double __x) throw(); 
# 163
extern long double hypotl(long double __x, long double __y) throw(); extern long double __hypotl(long double __x, long double __y) throw(); 
# 170
extern long double cbrtl(long double __x) throw(); extern long double __cbrtl(long double __x) throw(); 
# 179
extern long double ceill(long double __x) throw() __attribute((const)); extern long double __ceill(long double __x) throw() __attribute((const)); 
# 182
extern long double fabsl(long double __x) throw() __attribute((const)); extern long double __fabsl(long double __x) throw() __attribute((const)); 
# 185
extern long double floorl(long double __x) throw() __attribute((const)); extern long double __floorl(long double __x) throw() __attribute((const)); 
# 188
extern long double fmodl(long double __x, long double __y) throw(); extern long double __fmodl(long double __x, long double __y) throw(); 
# 193
extern int __isinfl(long double __value) throw() __attribute((const)); 
# 196
extern int __finitel(long double __value) throw() __attribute((const)); 
# 202
extern int isinfl(long double __value) throw() __attribute((const)); 
# 205
extern int finitel(long double __value) throw() __attribute((const)); 
# 208
extern long double dreml(long double __x, long double __y) throw(); extern long double __dreml(long double __x, long double __y) throw(); 
# 212
extern long double significandl(long double __x) throw(); extern long double __significandl(long double __x) throw(); 
# 218
extern long double copysignl(long double __x, long double __y) throw() __attribute((const)); extern long double __copysignl(long double __x, long double __y) throw() __attribute((const)); 
# 225
extern long double nanl(const char * __tagb) throw() __attribute((const)); extern long double __nanl(const char * __tagb) throw() __attribute((const)); 
# 231
extern int __isnanl(long double __value) throw() __attribute((const)); 
# 235
extern int isnanl(long double __value) throw() __attribute((const)); 
# 238
extern long double j0l(long double) throw(); extern long double __j0l(long double) throw(); 
# 239
extern long double j1l(long double) throw(); extern long double __j1l(long double) throw(); 
# 240
extern long double jnl(int, long double) throw(); extern long double __jnl(int, long double) throw(); 
# 241
extern long double y0l(long double) throw(); extern long double __y0l(long double) throw(); 
# 242
extern long double y1l(long double) throw(); extern long double __y1l(long double) throw(); 
# 243
extern long double ynl(int, long double) throw(); extern long double __ynl(int, long double) throw(); 
# 250
extern long double erfl(long double) throw(); extern long double __erfl(long double) throw(); 
# 251
extern long double erfcl(long double) throw(); extern long double __erfcl(long double) throw(); 
# 252
extern long double lgammal(long double) throw(); extern long double __lgammal(long double) throw(); 
# 259
extern long double tgammal(long double) throw(); extern long double __tgammal(long double) throw(); 
# 265
extern long double gammal(long double) throw(); extern long double __gammal(long double) throw(); 
# 272
extern long double lgammal_r(long double, int * __signgamp) throw(); extern long double __lgammal_r(long double, int * __signgamp) throw(); 
# 280
extern long double rintl(long double __x) throw(); extern long double __rintl(long double __x) throw(); 
# 283
extern long double nextafterl(long double __x, long double __y) throw() __attribute((const)); extern long double __nextafterl(long double __x, long double __y) throw() __attribute((const)); 
# 285
extern long double nexttowardl(long double __x, long double __y) throw() __attribute((const)); extern long double __nexttowardl(long double __x, long double __y) throw() __attribute((const)); 
# 289
extern long double remainderl(long double __x, long double __y) throw(); extern long double __remainderl(long double __x, long double __y) throw(); 
# 293
extern long double scalbnl(long double __x, int __n) throw(); extern long double __scalbnl(long double __x, int __n) throw(); 
# 297
extern int ilogbl(long double __x) throw(); extern int __ilogbl(long double __x) throw(); 
# 302
extern long double scalblnl(long double __x, long __n) throw(); extern long double __scalblnl(long double __x, long __n) throw(); 
# 306
extern long double nearbyintl(long double __x) throw(); extern long double __nearbyintl(long double __x) throw(); 
# 310
extern long double roundl(long double __x) throw() __attribute((const)); extern long double __roundl(long double __x) throw() __attribute((const)); 
# 314
extern long double truncl(long double __x) throw() __attribute((const)); extern long double __truncl(long double __x) throw() __attribute((const)); 
# 319
extern long double remquol(long double __x, long double __y, int * __quo) throw(); extern long double __remquol(long double __x, long double __y, int * __quo) throw(); 
# 326
extern long lrintl(long double __x) throw(); extern long __lrintl(long double __x) throw(); 
# 327
extern long long llrintl(long double __x) throw(); extern long long __llrintl(long double __x) throw(); 
# 331
extern long lroundl(long double __x) throw(); extern long __lroundl(long double __x) throw(); 
# 332
extern long long llroundl(long double __x) throw(); extern long long __llroundl(long double __x) throw(); 
# 336
extern long double fdiml(long double __x, long double __y) throw(); extern long double __fdiml(long double __x, long double __y) throw(); 
# 339
extern long double fmaxl(long double __x, long double __y) throw() __attribute((const)); extern long double __fmaxl(long double __x, long double __y) throw() __attribute((const)); 
# 342
extern long double fminl(long double __x, long double __y) throw() __attribute((const)); extern long double __fminl(long double __x, long double __y) throw() __attribute((const)); 
# 346
extern int __fpclassifyl(long double __value) throw()
# 347
 __attribute((const)); 
# 350
extern int __signbitl(long double __value) throw()
# 351
 __attribute((const)); 
# 355
extern long double fmal(long double __x, long double __y, long double __z) throw(); extern long double __fmal(long double __x, long double __y, long double __z) throw(); 
# 364
extern long double scalbl(long double __x, long double __n) throw(); extern long double __scalbl(long double __x, long double __n) throw(); 
# 149 "/usr/include/math.h" 3
extern int signgam; 
# 191 "/usr/include/math.h" 3
enum { 
# 192
FP_NAN, 
# 195
FP_INFINITE, 
# 198
FP_ZERO, 
# 201
FP_SUBNORMAL, 
# 204
FP_NORMAL
# 207
}; 
# 295 "/usr/include/math.h" 3
typedef 
# 289
enum { 
# 290
_IEEE_ = (-1), 
# 291
_SVID_ = 0, 
# 292
_XOPEN_, 
# 293
_POSIX_, 
# 294
_ISOC_
# 295
} _LIB_VERSION_TYPE; 
# 300
extern _LIB_VERSION_TYPE _LIB_VERSION; 
# 311 "/usr/include/math.h" 3
struct __exception { 
# 316
int type; 
# 317
char *name; 
# 318
double arg1; 
# 319
double arg2; 
# 320
double retval; 
# 321
}; 
# 324
extern int matherr(__exception * __exc) throw(); 
# 475 "/usr/include/math.h" 3
}
# 34 "/usr/include/stdlib.h" 3
extern "C" {
# 45 "/usr/include/bits/byteswap.h" 3
static inline unsigned __bswap_32(unsigned __bsx) 
# 46
{ 
# 47
return __builtin_bswap32(__bsx); 
# 48
} 
# 109 "/usr/include/bits/byteswap.h" 3
static inline __uint64_t __bswap_64(__uint64_t __bsx) 
# 110
{ 
# 111
return __builtin_bswap64(__bsx); 
# 112
} 
# 66 "/usr/include/bits/waitstatus.h" 3
union wait { 
# 68
int w_status; 
# 70
struct { 
# 72
unsigned __w_termsig:7; 
# 73
unsigned __w_coredump:1; 
# 74
unsigned __w_retcode:8; 
# 75
unsigned:16; 
# 83
} __wait_terminated; 
# 85
struct { 
# 87
unsigned __w_stopval:8; 
# 88
unsigned __w_stopsig:8; 
# 89
unsigned:16; 
# 96
} __wait_stopped; 
# 97
}; 
# 101 "/usr/include/stdlib.h" 3
typedef 
# 98
struct { 
# 99
int quot; 
# 100
int rem; 
# 101
} div_t; 
# 109
typedef 
# 106
struct { 
# 107
long quot; 
# 108
long rem; 
# 109
} ldiv_t; 
# 121
__extension__ typedef 
# 118
struct { 
# 119
long long quot; 
# 120
long long rem; 
# 121
} lldiv_t; 
# 139 "/usr/include/stdlib.h" 3
extern size_t __ctype_get_mb_cur_max() throw(); 
# 144
extern double atof(const char * __nptr) throw()
# 145
 __attribute((__pure__)) __attribute((__nonnull__(1))); 
# 147
extern int atoi(const char * __nptr) throw()
# 148
 __attribute((__pure__)) __attribute((__nonnull__(1))); 
# 150
extern long atol(const char * __nptr) throw()
# 151
 __attribute((__pure__)) __attribute((__nonnull__(1))); 
# 157
__extension__ extern long long atoll(const char * __nptr) throw()
# 158
 __attribute((__pure__)) __attribute((__nonnull__(1))); 
# 164
extern double strtod(const char *__restrict__ __nptr, char **__restrict__ __endptr) throw()
# 166
 __attribute((__nonnull__(1))); 
# 172
extern float strtof(const char *__restrict__ __nptr, char **__restrict__ __endptr) throw()
# 173
 __attribute((__nonnull__(1))); 
# 175
extern long double strtold(const char *__restrict__ __nptr, char **__restrict__ __endptr) throw()
# 177
 __attribute((__nonnull__(1))); 
# 183
extern long strtol(const char *__restrict__ __nptr, char **__restrict__ __endptr, int __base) throw()
# 185
 __attribute((__nonnull__(1))); 
# 187
extern unsigned long strtoul(const char *__restrict__ __nptr, char **__restrict__ __endptr, int __base) throw()
# 189
 __attribute((__nonnull__(1))); 
# 195
__extension__ extern long long strtoq(const char *__restrict__ __nptr, char **__restrict__ __endptr, int __base) throw()
# 197
 __attribute((__nonnull__(1))); 
# 200
__extension__ extern unsigned long long strtouq(const char *__restrict__ __nptr, char **__restrict__ __endptr, int __base) throw()
# 202
 __attribute((__nonnull__(1))); 
# 209
__extension__ extern long long strtoll(const char *__restrict__ __nptr, char **__restrict__ __endptr, int __base) throw()
# 211
 __attribute((__nonnull__(1))); 
# 214
__extension__ extern unsigned long long strtoull(const char *__restrict__ __nptr, char **__restrict__ __endptr, int __base) throw()
# 216
 __attribute((__nonnull__(1))); 
# 239 "/usr/include/stdlib.h" 3
extern long strtol_l(const char *__restrict__ __nptr, char **__restrict__ __endptr, int __base, __locale_t __loc) throw()
# 241
 __attribute((__nonnull__(1, 4))); 
# 243
extern unsigned long strtoul_l(const char *__restrict__ __nptr, char **__restrict__ __endptr, int __base, __locale_t __loc) throw()
# 246
 __attribute((__nonnull__(1, 4))); 
# 249
__extension__ extern long long strtoll_l(const char *__restrict__ __nptr, char **__restrict__ __endptr, int __base, __locale_t __loc) throw()
# 252
 __attribute((__nonnull__(1, 4))); 
# 255
__extension__ extern unsigned long long strtoull_l(const char *__restrict__ __nptr, char **__restrict__ __endptr, int __base, __locale_t __loc) throw()
# 258
 __attribute((__nonnull__(1, 4))); 
# 260
extern double strtod_l(const char *__restrict__ __nptr, char **__restrict__ __endptr, __locale_t __loc) throw()
# 262
 __attribute((__nonnull__(1, 3))); 
# 264
extern float strtof_l(const char *__restrict__ __nptr, char **__restrict__ __endptr, __locale_t __loc) throw()
# 266
 __attribute((__nonnull__(1, 3))); 
# 268
extern long double strtold_l(const char *__restrict__ __nptr, char **__restrict__ __endptr, __locale_t __loc) throw()
# 271
 __attribute((__nonnull__(1, 3))); 
# 305 "/usr/include/stdlib.h" 3
extern char *l64a(long __n) throw(); 
# 308
extern long a64l(const char * __s) throw()
# 309
 __attribute((__pure__)) __attribute((__nonnull__(1))); 
# 27 "/usr/include/sys/types.h" 3
extern "C" {
# 33
typedef __u_char u_char; 
# 34
typedef __u_short u_short; 
# 35
typedef __u_int u_int; 
# 36
typedef __u_long u_long; 
# 37
typedef __quad_t quad_t; 
# 38
typedef __u_quad_t u_quad_t; 
# 39
typedef __fsid_t fsid_t; 
# 44
typedef __loff_t loff_t; 
# 48
typedef __ino_t ino_t; 
# 55
typedef __ino64_t ino64_t; 
# 60
typedef __dev_t dev_t; 
# 65
typedef __gid_t gid_t; 
# 70
typedef __mode_t mode_t; 
# 75
typedef __nlink_t nlink_t; 
# 80
typedef __uid_t uid_t; 
# 86
typedef __off_t off_t; 
# 93
typedef __off64_t off64_t; 
# 104 "/usr/include/sys/types.h" 3
typedef __id_t id_t; 
# 109
typedef __ssize_t ssize_t; 
# 115
typedef __daddr_t daddr_t; 
# 116
typedef __caddr_t caddr_t; 
# 122
typedef __key_t key_t; 
# 136 "/usr/include/sys/types.h" 3
typedef __useconds_t useconds_t; 
# 140
typedef __suseconds_t suseconds_t; 
# 150 "/usr/include/sys/types.h" 3
typedef unsigned long ulong; 
# 151
typedef unsigned short ushort; 
# 152
typedef unsigned uint; 
# 194 "/usr/include/sys/types.h" 3
typedef signed char int8_t __attribute((__mode__(__QI__))); 
# 195
typedef short int16_t __attribute((__mode__(__HI__))); 
# 196
typedef int int32_t __attribute((__mode__(__SI__))); 
# 197
typedef long int64_t __attribute((__mode__(__DI__))); 
# 200
typedef unsigned char u_int8_t __attribute((__mode__(__QI__))); 
# 201
typedef unsigned short u_int16_t __attribute((__mode__(__HI__))); 
# 202
typedef unsigned u_int32_t __attribute((__mode__(__SI__))); 
# 203
typedef unsigned long u_int64_t __attribute((__mode__(__DI__))); 
# 205
typedef long register_t __attribute((__mode__(__word__))); 
# 23 "/usr/include/bits/sigset.h" 3
typedef int __sig_atomic_t; 
# 31
typedef 
# 29
struct { 
# 30
unsigned long __val[(1024) / ((8) * sizeof(unsigned long))]; 
# 31
} __sigset_t; 
# 37 "/usr/include/sys/select.h" 3
typedef __sigset_t sigset_t; 
# 54 "/usr/include/sys/select.h" 3
typedef long __fd_mask; 
# 75 "/usr/include/sys/select.h" 3
typedef 
# 65
struct { 
# 69
__fd_mask fds_bits[1024 / (8 * ((int)sizeof(__fd_mask)))]; 
# 75
} fd_set; 
# 82
typedef __fd_mask fd_mask; 
# 96 "/usr/include/sys/select.h" 3
extern "C" {
# 106 "/usr/include/sys/select.h" 3
extern int select(int __nfds, fd_set *__restrict__ __readfds, fd_set *__restrict__ __writefds, fd_set *__restrict__ __exceptfds, timeval *__restrict__ __timeout); 
# 118 "/usr/include/sys/select.h" 3
extern int pselect(int __nfds, fd_set *__restrict__ __readfds, fd_set *__restrict__ __writefds, fd_set *__restrict__ __exceptfds, const timespec *__restrict__ __timeout, const __sigset_t *__restrict__ __sigmask); 
# 131 "/usr/include/sys/select.h" 3
}
# 29 "/usr/include/sys/sysmacros.h" 3
extern "C" {
# 32
__extension__ extern unsigned gnu_dev_major(unsigned long long __dev) throw()
# 33
 __attribute((const)); 
# 35
__extension__ extern unsigned gnu_dev_minor(unsigned long long __dev) throw()
# 36
 __attribute((const)); 
# 38
__extension__ extern unsigned long long gnu_dev_makedev(unsigned __major, unsigned __minor) throw()
# 40
 __attribute((const)); 
# 63 "/usr/include/sys/sysmacros.h" 3
}
# 228 "/usr/include/sys/types.h" 3
typedef __blksize_t blksize_t; 
# 235
typedef __blkcnt_t blkcnt_t; 
# 239
typedef __fsblkcnt_t fsblkcnt_t; 
# 243
typedef __fsfilcnt_t fsfilcnt_t; 
# 262 "/usr/include/sys/types.h" 3
typedef __blkcnt64_t blkcnt64_t; 
# 263
typedef __fsblkcnt64_t fsblkcnt64_t; 
# 264
typedef __fsfilcnt64_t fsfilcnt64_t; 
# 60 "/usr/include/bits/pthreadtypes.h" 3
typedef unsigned long pthread_t; 
# 63
union pthread_attr_t { 
# 65
char __size[56]; 
# 66
long __align; 
# 67
}; 
# 69
typedef pthread_attr_t pthread_attr_t; 
# 79
typedef 
# 75
struct __pthread_internal_list { 
# 77
__pthread_internal_list *__prev; 
# 78
__pthread_internal_list *__next; 
# 79
} __pthread_list_t; 
# 127 "/usr/include/bits/pthreadtypes.h" 3
typedef 
# 91 "/usr/include/bits/pthreadtypes.h" 3
union { 
# 92
struct __pthread_mutex_s { 
# 94
int __lock; 
# 95
unsigned __count; 
# 96
int __owner; 
# 98
unsigned __nusers; 
# 102
int __kind; 
# 104
short __spins; 
# 105
short __elision; 
# 106
__pthread_list_t __list; 
# 124 "/usr/include/bits/pthreadtypes.h" 3
} __data; 
# 125
char __size[40]; 
# 126
long __align; 
# 127
} pthread_mutex_t; 
# 133
typedef 
# 130
union { 
# 131
char __size[4]; 
# 132
int __align; 
# 133
} pthread_mutexattr_t; 
# 153
typedef 
# 139
union { 
# 141
struct { 
# 142
int __lock; 
# 143
unsigned __futex; 
# 144
__extension__ unsigned long long __total_seq; 
# 145
__extension__ unsigned long long __wakeup_seq; 
# 146
__extension__ unsigned long long __woken_seq; 
# 147
void *__mutex; 
# 148
unsigned __nwaiters; 
# 149
unsigned __broadcast_seq; 
# 150
} __data; 
# 151
char __size[48]; 
# 152
__extension__ long long __align; 
# 153
} pthread_cond_t; 
# 159
typedef 
# 156
union { 
# 157
char __size[4]; 
# 158
int __align; 
# 159
} pthread_condattr_t; 
# 163
typedef unsigned pthread_key_t; 
# 167
typedef int pthread_once_t; 
# 213 "/usr/include/bits/pthreadtypes.h" 3
typedef 
# 174 "/usr/include/bits/pthreadtypes.h" 3
union { 
# 177
struct { 
# 178
int __lock; 
# 179
unsigned __nr_readers; 
# 180
unsigned __readers_wakeup; 
# 181
unsigned __writer_wakeup; 
# 182
unsigned __nr_readers_queued; 
# 183
unsigned __nr_writers_queued; 
# 184
int __writer; 
# 185
int __shared; 
# 186
unsigned long __pad1; 
# 187
unsigned long __pad2; 
# 190
unsigned __flags; 
# 192
} __data; 
# 211 "/usr/include/bits/pthreadtypes.h" 3
char __size[56]; 
# 212
long __align; 
# 213
} pthread_rwlock_t; 
# 219
typedef 
# 216
union { 
# 217
char __size[8]; 
# 218
long __align; 
# 219
} pthread_rwlockattr_t; 
# 225
typedef volatile int pthread_spinlock_t; 
# 234
typedef 
# 231
union { 
# 232
char __size[32]; 
# 233
long __align; 
# 234
} pthread_barrier_t; 
# 240
typedef 
# 237
union { 
# 238
char __size[4]; 
# 239
int __align; 
# 240
} pthread_barrierattr_t; 
# 273 "/usr/include/sys/types.h" 3
}
# 321 "/usr/include/stdlib.h" 3
extern long random() throw(); 
# 324
extern void srandom(unsigned __seed) throw(); 
# 330
extern char *initstate(unsigned __seed, char * __statebuf, size_t __statelen) throw()
# 331
 __attribute((__nonnull__(2))); 
# 335
extern char *setstate(char * __statebuf) throw() __attribute((__nonnull__(1))); 
# 343
struct random_data { 
# 345
int32_t *fptr; 
# 346
int32_t *rptr; 
# 347
int32_t *state; 
# 348
int rand_type; 
# 349
int rand_deg; 
# 350
int rand_sep; 
# 351
int32_t *end_ptr; 
# 352
}; 
# 354
extern int random_r(random_data *__restrict__ __buf, int32_t *__restrict__ __result) throw()
# 355
 __attribute((__nonnull__(1, 2))); 
# 357
extern int srandom_r(unsigned __seed, random_data * __buf) throw()
# 358
 __attribute((__nonnull__(2))); 
# 360
extern int initstate_r(unsigned __seed, char *__restrict__ __statebuf, size_t __statelen, random_data *__restrict__ __buf) throw()
# 363
 __attribute((__nonnull__(2, 4))); 
# 365
extern int setstate_r(char *__restrict__ __statebuf, random_data *__restrict__ __buf) throw()
# 367
 __attribute((__nonnull__(1, 2))); 
# 374
extern int rand() throw(); 
# 376
extern void srand(unsigned __seed) throw(); 
# 381
extern int rand_r(unsigned * __seed) throw(); 
# 389
extern double drand48() throw(); 
# 390
extern double erand48(unsigned short  __xsubi[3]) throw() __attribute((__nonnull__(1))); 
# 393
extern long lrand48() throw(); 
# 394
extern long nrand48(unsigned short  __xsubi[3]) throw()
# 395
 __attribute((__nonnull__(1))); 
# 398
extern long mrand48() throw(); 
# 399
extern long jrand48(unsigned short  __xsubi[3]) throw()
# 400
 __attribute((__nonnull__(1))); 
# 403
extern void srand48(long __seedval) throw(); 
# 404
extern unsigned short *seed48(unsigned short  __seed16v[3]) throw()
# 405
 __attribute((__nonnull__(1))); 
# 406
extern void lcong48(unsigned short  __param[7]) throw() __attribute((__nonnull__(1))); 
# 412
struct drand48_data { 
# 414
unsigned short __x[3]; 
# 415
unsigned short __old_x[3]; 
# 416
unsigned short __c; 
# 417
unsigned short __init; 
# 418
unsigned long long __a; 
# 419
}; 
# 422
extern int drand48_r(drand48_data *__restrict__ __buffer, double *__restrict__ __result) throw()
# 423
 __attribute((__nonnull__(1, 2))); 
# 424
extern int erand48_r(unsigned short  __xsubi[3], drand48_data *__restrict__ __buffer, double *__restrict__ __result) throw()
# 426
 __attribute((__nonnull__(1, 2))); 
# 429
extern int lrand48_r(drand48_data *__restrict__ __buffer, long *__restrict__ __result) throw()
# 431
 __attribute((__nonnull__(1, 2))); 
# 432
extern int nrand48_r(unsigned short  __xsubi[3], drand48_data *__restrict__ __buffer, long *__restrict__ __result) throw()
# 435
 __attribute((__nonnull__(1, 2))); 
# 438
extern int mrand48_r(drand48_data *__restrict__ __buffer, long *__restrict__ __result) throw()
# 440
 __attribute((__nonnull__(1, 2))); 
# 441
extern int jrand48_r(unsigned short  __xsubi[3], drand48_data *__restrict__ __buffer, long *__restrict__ __result) throw()
# 444
 __attribute((__nonnull__(1, 2))); 
# 447
extern int srand48_r(long __seedval, drand48_data * __buffer) throw()
# 448
 __attribute((__nonnull__(2))); 
# 450
extern int seed48_r(unsigned short  __seed16v[3], drand48_data * __buffer) throw()
# 451
 __attribute((__nonnull__(1, 2))); 
# 453
extern int lcong48_r(unsigned short  __param[7], drand48_data * __buffer) throw()
# 455
 __attribute((__nonnull__(1, 2))); 
# 465
extern void *malloc(size_t __size) throw() __attribute((__malloc__)); 
# 467
extern void *calloc(size_t __nmemb, size_t __size) throw()
# 468
 __attribute((__malloc__)); 
# 479
extern void *realloc(void * __ptr, size_t __size) throw()
# 480
 __attribute((__warn_unused_result__)); 
# 482
extern void free(void * __ptr) throw(); 
# 487
extern void cfree(void * __ptr) throw(); 
# 26 "/usr/include/alloca.h" 3
extern "C" {
# 32
extern void *alloca(size_t __size) throw(); 
# 38
}
# 497 "/usr/include/stdlib.h" 3
extern void *valloc(size_t __size) throw() __attribute((__malloc__)); 
# 502
extern int posix_memalign(void ** __memptr, size_t __alignment, size_t __size) throw()
# 503
 __attribute((__nonnull__(1))); 
# 508
extern void *aligned_alloc(size_t __alignment, size_t __size) throw()
# 509
 __attribute((__malloc__, __alloc_size__(2))); 
# 514
extern void abort() throw() __attribute((__noreturn__)); 
# 518
extern int atexit(void (* __func)(void)) throw() __attribute((__nonnull__(1))); 
# 523
extern "C++" int at_quick_exit(void (* __func)(void)) throw() __asm__("at_quick_exit")
# 524
 __attribute((__nonnull__(1))); 
# 534
extern int on_exit(void (* __func)(int __status, void * __arg), void * __arg) throw()
# 535
 __attribute((__nonnull__(1))); 
# 542
extern void exit(int __status) throw() __attribute((__noreturn__)); 
# 548
extern void quick_exit(int __status) throw() __attribute((__noreturn__)); 
# 556
extern void _Exit(int __status) throw() __attribute((__noreturn__)); 
# 563
extern char *getenv(const char * __name) throw() __attribute((__nonnull__(1))); 
# 569
extern char *secure_getenv(const char * __name) throw()
# 570
 __attribute((__nonnull__(1))); 
# 577
extern int putenv(char * __string) throw() __attribute((__nonnull__(1))); 
# 583
extern int setenv(const char * __name, const char * __value, int __replace) throw()
# 584
 __attribute((__nonnull__(2))); 
# 587
extern int unsetenv(const char * __name) throw() __attribute((__nonnull__(1))); 
# 594
extern int clearenv() throw(); 
# 605 "/usr/include/stdlib.h" 3
extern char *mktemp(char * __template) throw() __attribute((__nonnull__(1))); 
# 619 "/usr/include/stdlib.h" 3
extern int mkstemp(char * __template) __attribute((__nonnull__(1))); 
# 629 "/usr/include/stdlib.h" 3
extern int mkstemp64(char * __template) __attribute((__nonnull__(1))); 
# 641 "/usr/include/stdlib.h" 3
extern int mkstemps(char * __template, int __suffixlen) __attribute((__nonnull__(1))); 
# 651 "/usr/include/stdlib.h" 3
extern int mkstemps64(char * __template, int __suffixlen)
# 652
 __attribute((__nonnull__(1))); 
# 662 "/usr/include/stdlib.h" 3
extern char *mkdtemp(char * __template) throw() __attribute((__nonnull__(1))); 
# 673 "/usr/include/stdlib.h" 3
extern int mkostemp(char * __template, int __flags) __attribute((__nonnull__(1))); 
# 683 "/usr/include/stdlib.h" 3
extern int mkostemp64(char * __template, int __flags) __attribute((__nonnull__(1))); 
# 693 "/usr/include/stdlib.h" 3
extern int mkostemps(char * __template, int __suffixlen, int __flags)
# 694
 __attribute((__nonnull__(1))); 
# 705 "/usr/include/stdlib.h" 3
extern int mkostemps64(char * __template, int __suffixlen, int __flags)
# 706
 __attribute((__nonnull__(1))); 
# 716
extern int system(const char * __command); 
# 723
extern char *canonicalize_file_name(const char * __name) throw()
# 724
 __attribute((__nonnull__(1))); 
# 733 "/usr/include/stdlib.h" 3
extern char *realpath(const char *__restrict__ __name, char *__restrict__ __resolved) throw(); 
# 741
typedef int (*__compar_fn_t)(const void *, const void *); 
# 744
typedef __compar_fn_t comparison_fn_t; 
# 748
typedef int (*__compar_d_fn_t)(const void *, const void *, void *); 
# 754
extern void *bsearch(const void * __key, const void * __base, size_t __nmemb, size_t __size, __compar_fn_t __compar)
# 756
 __attribute((__nonnull__(1, 2, 5))); 
# 760
extern void qsort(void * __base, size_t __nmemb, size_t __size, __compar_fn_t __compar)
# 761
 __attribute((__nonnull__(1, 4))); 
# 763
extern void qsort_r(void * __base, size_t __nmemb, size_t __size, __compar_d_fn_t __compar, void * __arg)
# 765
 __attribute((__nonnull__(1, 4))); 
# 770
extern int abs(int __x) throw() __attribute((const)); 
# 771
extern long labs(long __x) throw() __attribute((const)); 
# 775
__extension__ extern long long llabs(long long __x) throw()
# 776
 __attribute((const)); 
# 784
extern div_t div(int __numer, int __denom) throw()
# 785
 __attribute((const)); 
# 786
extern ldiv_t ldiv(long __numer, long __denom) throw()
# 787
 __attribute((const)); 
# 792
__extension__ extern lldiv_t lldiv(long long __numer, long long __denom) throw()
# 794
 __attribute((const)); 
# 807 "/usr/include/stdlib.h" 3
extern char *ecvt(double __value, int __ndigit, int *__restrict__ __decpt, int *__restrict__ __sign) throw()
# 808
 __attribute((__nonnull__(3, 4))); 
# 813
extern char *fcvt(double __value, int __ndigit, int *__restrict__ __decpt, int *__restrict__ __sign) throw()
# 814
 __attribute((__nonnull__(3, 4))); 
# 819
extern char *gcvt(double __value, int __ndigit, char * __buf) throw()
# 820
 __attribute((__nonnull__(3))); 
# 825
extern char *qecvt(long double __value, int __ndigit, int *__restrict__ __decpt, int *__restrict__ __sign) throw()
# 827
 __attribute((__nonnull__(3, 4))); 
# 828
extern char *qfcvt(long double __value, int __ndigit, int *__restrict__ __decpt, int *__restrict__ __sign) throw()
# 830
 __attribute((__nonnull__(3, 4))); 
# 831
extern char *qgcvt(long double __value, int __ndigit, char * __buf) throw()
# 832
 __attribute((__nonnull__(3))); 
# 837
extern int ecvt_r(double __value, int __ndigit, int *__restrict__ __decpt, int *__restrict__ __sign, char *__restrict__ __buf, size_t __len) throw()
# 839
 __attribute((__nonnull__(3, 4, 5))); 
# 840
extern int fcvt_r(double __value, int __ndigit, int *__restrict__ __decpt, int *__restrict__ __sign, char *__restrict__ __buf, size_t __len) throw()
# 842
 __attribute((__nonnull__(3, 4, 5))); 
# 844
extern int qecvt_r(long double __value, int __ndigit, int *__restrict__ __decpt, int *__restrict__ __sign, char *__restrict__ __buf, size_t __len) throw()
# 847
 __attribute((__nonnull__(3, 4, 5))); 
# 848
extern int qfcvt_r(long double __value, int __ndigit, int *__restrict__ __decpt, int *__restrict__ __sign, char *__restrict__ __buf, size_t __len) throw()
# 851
 __attribute((__nonnull__(3, 4, 5))); 
# 859
extern int mblen(const char * __s, size_t __n) throw(); 
# 862
extern int mbtowc(wchar_t *__restrict__ __pwc, const char *__restrict__ __s, size_t __n) throw(); 
# 866
extern int wctomb(char * __s, wchar_t __wchar) throw(); 
# 870
extern size_t mbstowcs(wchar_t *__restrict__ __pwcs, const char *__restrict__ __s, size_t __n) throw(); 
# 873
extern size_t wcstombs(char *__restrict__ __s, const wchar_t *__restrict__ __pwcs, size_t __n) throw(); 
# 884
extern int rpmatch(const char * __response) throw() __attribute((__nonnull__(1))); 
# 895 "/usr/include/stdlib.h" 3
extern int getsubopt(char **__restrict__ __optionp, char *const *__restrict__ __tokens, char **__restrict__ __valuep) throw()
# 898
 __attribute((__nonnull__(1, 2, 3))); 
# 904
extern void setkey(const char * __key) throw() __attribute((__nonnull__(1))); 
# 912
extern int posix_openpt(int __oflag); 
# 920
extern int grantpt(int __fd) throw(); 
# 924
extern int unlockpt(int __fd) throw(); 
# 929
extern char *ptsname(int __fd) throw(); 
# 936
extern int ptsname_r(int __fd, char * __buf, size_t __buflen) throw()
# 937
 __attribute((__nonnull__(2))); 
# 940
extern int getpt(); 
# 947
extern int getloadavg(double  __loadavg[], int __nelem) throw()
# 948
 __attribute((__nonnull__(1))); 
# 964 "/usr/include/stdlib.h" 3
}
# 46 "/opt/rh/devtoolset-7/root/usr/include/c++/7/bits/std_abs.h" 3
extern "C++" {
# 48
namespace std __attribute((__visibility__("default"))) { 
# 52
using ::abs;
# 56
inline long abs(long __i) { return __builtin_labs(__i); } 
# 61
inline long long abs(long long __x) { return __builtin_llabs(__x); } 
# 70
constexpr double abs(double __x) 
# 71
{ return __builtin_fabs(__x); } 
# 74
constexpr float abs(float __x) 
# 75
{ return __builtin_fabsf(__x); } 
# 78
constexpr long double abs(long double __x) 
# 79
{ return __builtin_fabsl(__x); } 
# 84
constexpr __int128 abs(__int128 __x) { return (__x >= (0)) ? __x : (-__x); } 
# 102 "/opt/rh/devtoolset-7/root/usr/include/c++/7/bits/std_abs.h" 3
constexpr __float128 abs(__float128 __x) 
# 103
{ return (__x < (0)) ? -__x : __x; } 
# 107
}
# 108
}
# 77 "/opt/rh/devtoolset-7/root/usr/include/c++/7/cmath" 3
extern "C++" {
# 79
namespace std __attribute((__visibility__("default"))) { 
# 83
using ::acos;
# 87
constexpr float acos(float __x) 
# 88
{ return __builtin_acosf(__x); } 
# 91
constexpr long double acos(long double __x) 
# 92
{ return __builtin_acosl(__x); } 
# 95
template< class _Tp> constexpr typename __gnu_cxx::__enable_if< __is_integer< _Tp> ::__value, double> ::__type 
# 99
acos(_Tp __x) 
# 100
{ return __builtin_acos(__x); } 
# 102
using ::asin;
# 106
constexpr float asin(float __x) 
# 107
{ return __builtin_asinf(__x); } 
# 110
constexpr long double asin(long double __x) 
# 111
{ return __builtin_asinl(__x); } 
# 114
template< class _Tp> constexpr typename __gnu_cxx::__enable_if< __is_integer< _Tp> ::__value, double> ::__type 
# 118
asin(_Tp __x) 
# 119
{ return __builtin_asin(__x); } 
# 121
using ::atan;
# 125
constexpr float atan(float __x) 
# 126
{ return __builtin_atanf(__x); } 
# 129
constexpr long double atan(long double __x) 
# 130
{ return __builtin_atanl(__x); } 
# 133
template< class _Tp> constexpr typename __gnu_cxx::__enable_if< __is_integer< _Tp> ::__value, double> ::__type 
# 137
atan(_Tp __x) 
# 138
{ return __builtin_atan(__x); } 
# 140
using ::atan2;
# 144
constexpr float atan2(float __y, float __x) 
# 145
{ return __builtin_atan2f(__y, __x); } 
# 148
constexpr long double atan2(long double __y, long double __x) 
# 149
{ return __builtin_atan2l(__y, __x); } 
# 152
template< class _Tp, class _Up> constexpr typename __gnu_cxx::__promote_2< _Tp, _Up> ::__type 
# 155
atan2(_Tp __y, _Up __x) 
# 156
{ 
# 157
typedef typename __gnu_cxx::__promote_2< _Tp, _Up> ::__type __type; 
# 158
return atan2((__type)__y, (__type)__x); 
# 159
} 
# 161
using ::ceil;
# 165
constexpr float ceil(float __x) 
# 166
{ return __builtin_ceilf(__x); } 
# 169
constexpr long double ceil(long double __x) 
# 170
{ return __builtin_ceill(__x); } 
# 173
template< class _Tp> constexpr typename __gnu_cxx::__enable_if< __is_integer< _Tp> ::__value, double> ::__type 
# 177
ceil(_Tp __x) 
# 178
{ return __builtin_ceil(__x); } 
# 180
using ::cos;
# 184
constexpr float cos(float __x) 
# 185
{ return __builtin_cosf(__x); } 
# 188
constexpr long double cos(long double __x) 
# 189
{ return __builtin_cosl(__x); } 
# 192
template< class _Tp> constexpr typename __gnu_cxx::__enable_if< __is_integer< _Tp> ::__value, double> ::__type 
# 196
cos(_Tp __x) 
# 197
{ return __builtin_cos(__x); } 
# 199
using ::cosh;
# 203
constexpr float cosh(float __x) 
# 204
{ return __builtin_coshf(__x); } 
# 207
constexpr long double cosh(long double __x) 
# 208
{ return __builtin_coshl(__x); } 
# 211
template< class _Tp> constexpr typename __gnu_cxx::__enable_if< __is_integer< _Tp> ::__value, double> ::__type 
# 215
cosh(_Tp __x) 
# 216
{ return __builtin_cosh(__x); } 
# 218
using ::exp;
# 222
constexpr float exp(float __x) 
# 223
{ return __builtin_expf(__x); } 
# 226
constexpr long double exp(long double __x) 
# 227
{ return __builtin_expl(__x); } 
# 230
template< class _Tp> constexpr typename __gnu_cxx::__enable_if< __is_integer< _Tp> ::__value, double> ::__type 
# 234
exp(_Tp __x) 
# 235
{ return __builtin_exp(__x); } 
# 237
using ::fabs;
# 241
constexpr float fabs(float __x) 
# 242
{ return __builtin_fabsf(__x); } 
# 245
constexpr long double fabs(long double __x) 
# 246
{ return __builtin_fabsl(__x); } 
# 249
template< class _Tp> constexpr typename __gnu_cxx::__enable_if< __is_integer< _Tp> ::__value, double> ::__type 
# 253
fabs(_Tp __x) 
# 254
{ return __builtin_fabs(__x); } 
# 256
using ::floor;
# 260
constexpr float floor(float __x) 
# 261
{ return __builtin_floorf(__x); } 
# 264
constexpr long double floor(long double __x) 
# 265
{ return __builtin_floorl(__x); } 
# 268
template< class _Tp> constexpr typename __gnu_cxx::__enable_if< __is_integer< _Tp> ::__value, double> ::__type 
# 272
floor(_Tp __x) 
# 273
{ return __builtin_floor(__x); } 
# 275
using ::fmod;
# 279
constexpr float fmod(float __x, float __y) 
# 280
{ return __builtin_fmodf(__x, __y); } 
# 283
constexpr long double fmod(long double __x, long double __y) 
# 284
{ return __builtin_fmodl(__x, __y); } 
# 287
template< class _Tp, class _Up> constexpr typename __gnu_cxx::__promote_2< _Tp, _Up> ::__type 
# 290
fmod(_Tp __x, _Up __y) 
# 291
{ 
# 292
typedef typename __gnu_cxx::__promote_2< _Tp, _Up> ::__type __type; 
# 293
return fmod((__type)__x, (__type)__y); 
# 294
} 
# 296
using ::frexp;
# 300
inline float frexp(float __x, int *__exp) 
# 301
{ return __builtin_frexpf(__x, __exp); } 
# 304
inline long double frexp(long double __x, int *__exp) 
# 305
{ return __builtin_frexpl(__x, __exp); } 
# 308
template< class _Tp> constexpr typename __gnu_cxx::__enable_if< __is_integer< _Tp> ::__value, double> ::__type 
# 312
frexp(_Tp __x, int *__exp) 
# 313
{ return __builtin_frexp(__x, __exp); } 
# 315
using ::ldexp;
# 319
constexpr float ldexp(float __x, int __exp) 
# 320
{ return __builtin_ldexpf(__x, __exp); } 
# 323
constexpr long double ldexp(long double __x, int __exp) 
# 324
{ return __builtin_ldexpl(__x, __exp); } 
# 327
template< class _Tp> constexpr typename __gnu_cxx::__enable_if< __is_integer< _Tp> ::__value, double> ::__type 
# 331
ldexp(_Tp __x, int __exp) 
# 332
{ return __builtin_ldexp(__x, __exp); } 
# 334
using ::log;
# 338
constexpr float log(float __x) 
# 339
{ return __builtin_logf(__x); } 
# 342
constexpr long double log(long double __x) 
# 343
{ return __builtin_logl(__x); } 
# 346
template< class _Tp> constexpr typename __gnu_cxx::__enable_if< __is_integer< _Tp> ::__value, double> ::__type 
# 350
log(_Tp __x) 
# 351
{ return __builtin_log(__x); } 
# 353
using ::log10;
# 357
constexpr float log10(float __x) 
# 358
{ return __builtin_log10f(__x); } 
# 361
constexpr long double log10(long double __x) 
# 362
{ return __builtin_log10l(__x); } 
# 365
template< class _Tp> constexpr typename __gnu_cxx::__enable_if< __is_integer< _Tp> ::__value, double> ::__type 
# 369
log10(_Tp __x) 
# 370
{ return __builtin_log10(__x); } 
# 372
using ::modf;
# 376
inline float modf(float __x, float *__iptr) 
# 377
{ return __builtin_modff(__x, __iptr); } 
# 380
inline long double modf(long double __x, long double *__iptr) 
# 381
{ return __builtin_modfl(__x, __iptr); } 
# 384
using ::pow;
# 388
constexpr float pow(float __x, float __y) 
# 389
{ return __builtin_powf(__x, __y); } 
# 392
constexpr long double pow(long double __x, long double __y) 
# 393
{ return __builtin_powl(__x, __y); } 
# 412 "/opt/rh/devtoolset-7/root/usr/include/c++/7/cmath" 3
template< class _Tp, class _Up> constexpr typename __gnu_cxx::__promote_2< _Tp, _Up> ::__type 
# 415
pow(_Tp __x, _Up __y) 
# 416
{ 
# 417
typedef typename __gnu_cxx::__promote_2< _Tp, _Up> ::__type __type; 
# 418
return pow((__type)__x, (__type)__y); 
# 419
} 
# 421
using ::sin;
# 425
constexpr float sin(float __x) 
# 426
{ return __builtin_sinf(__x); } 
# 429
constexpr long double sin(long double __x) 
# 430
{ return __builtin_sinl(__x); } 
# 433
template< class _Tp> constexpr typename __gnu_cxx::__enable_if< __is_integer< _Tp> ::__value, double> ::__type 
# 437
sin(_Tp __x) 
# 438
{ return __builtin_sin(__x); } 
# 440
using ::sinh;
# 444
constexpr float sinh(float __x) 
# 445
{ return __builtin_sinhf(__x); } 
# 448
constexpr long double sinh(long double __x) 
# 449
{ return __builtin_sinhl(__x); } 
# 452
template< class _Tp> constexpr typename __gnu_cxx::__enable_if< __is_integer< _Tp> ::__value, double> ::__type 
# 456
sinh(_Tp __x) 
# 457
{ return __builtin_sinh(__x); } 
# 459
using ::sqrt;
# 463
constexpr float sqrt(float __x) 
# 464
{ return __builtin_sqrtf(__x); } 
# 467
constexpr long double sqrt(long double __x) 
# 468
{ return __builtin_sqrtl(__x); } 
# 471
template< class _Tp> constexpr typename __gnu_cxx::__enable_if< __is_integer< _Tp> ::__value, double> ::__type 
# 475
sqrt(_Tp __x) 
# 476
{ return __builtin_sqrt(__x); } 
# 478
using ::tan;
# 482
constexpr float tan(float __x) 
# 483
{ return __builtin_tanf(__x); } 
# 486
constexpr long double tan(long double __x) 
# 487
{ return __builtin_tanl(__x); } 
# 490
template< class _Tp> constexpr typename __gnu_cxx::__enable_if< __is_integer< _Tp> ::__value, double> ::__type 
# 494
tan(_Tp __x) 
# 495
{ return __builtin_tan(__x); } 
# 497
using ::tanh;
# 501
constexpr float tanh(float __x) 
# 502
{ return __builtin_tanhf(__x); } 
# 505
constexpr long double tanh(long double __x) 
# 506
{ return __builtin_tanhl(__x); } 
# 509
template< class _Tp> constexpr typename __gnu_cxx::__enable_if< __is_integer< _Tp> ::__value, double> ::__type 
# 513
tanh(_Tp __x) 
# 514
{ return __builtin_tanh(__x); } 
# 517
}
# 536 "/opt/rh/devtoolset-7/root/usr/include/c++/7/cmath" 3
namespace std __attribute((__visibility__("default"))) { 
# 544
constexpr int fpclassify(float __x) 
# 545
{ return __builtin_fpclassify(0, 1, 4, 3, 2, __x); 
# 546
} 
# 549
constexpr int fpclassify(double __x) 
# 550
{ return __builtin_fpclassify(0, 1, 4, 3, 2, __x); 
# 551
} 
# 554
constexpr int fpclassify(long double __x) 
# 555
{ return __builtin_fpclassify(0, 1, 4, 3, 2, __x); 
# 556
} 
# 560
template< class _Tp> constexpr typename __gnu_cxx::__enable_if< __is_integer< _Tp> ::__value, int> ::__type 
# 563
fpclassify(_Tp __x) 
# 564
{ return (__x != 0) ? 4 : 2; } 
# 569
constexpr bool isfinite(float __x) 
# 570
{ return __builtin_isfinite(__x); } 
# 573
constexpr bool isfinite(double __x) 
# 574
{ return __builtin_isfinite(__x); } 
# 577
constexpr bool isfinite(long double __x) 
# 578
{ return __builtin_isfinite(__x); } 
# 582
template< class _Tp> constexpr typename __gnu_cxx::__enable_if< __is_integer< _Tp> ::__value, bool> ::__type 
# 585
isfinite(_Tp __x) 
# 586
{ return true; } 
# 591
constexpr bool isinf(float __x) 
# 592
{ return __builtin_isinf(__x); } 
# 596
using ::isinf;
# 604
constexpr bool isinf(long double __x) 
# 605
{ return __builtin_isinf(__x); } 
# 609
template< class _Tp> constexpr typename __gnu_cxx::__enable_if< __is_integer< _Tp> ::__value, bool> ::__type 
# 612
isinf(_Tp __x) 
# 613
{ return false; } 
# 618
constexpr bool isnan(float __x) 
# 619
{ return __builtin_isnan(__x); } 
# 623
using ::isnan;
# 631
constexpr bool isnan(long double __x) 
# 632
{ return __builtin_isnan(__x); } 
# 636
template< class _Tp> constexpr typename __gnu_cxx::__enable_if< __is_integer< _Tp> ::__value, bool> ::__type 
# 639
isnan(_Tp __x) 
# 640
{ return false; } 
# 645
constexpr bool isnormal(float __x) 
# 646
{ return __builtin_isnormal(__x); } 
# 649
constexpr bool isnormal(double __x) 
# 650
{ return __builtin_isnormal(__x); } 
# 653
constexpr bool isnormal(long double __x) 
# 654
{ return __builtin_isnormal(__x); } 
# 658
template< class _Tp> constexpr typename __gnu_cxx::__enable_if< __is_integer< _Tp> ::__value, bool> ::__type 
# 661
isnormal(_Tp __x) 
# 662
{ return (__x != 0) ? true : false; } 
# 668
constexpr bool signbit(float __x) 
# 669
{ return __builtin_signbit(__x); } 
# 672
constexpr bool signbit(double __x) 
# 673
{ return __builtin_signbit(__x); } 
# 676
constexpr bool signbit(long double __x) 
# 677
{ return __builtin_signbit(__x); } 
# 681
template< class _Tp> constexpr typename __gnu_cxx::__enable_if< __is_integer< _Tp> ::__value, bool> ::__type 
# 684
signbit(_Tp __x) 
# 685
{ return (__x < 0) ? true : false; } 
# 690
constexpr bool isgreater(float __x, float __y) 
# 691
{ return __builtin_isgreater(__x, __y); } 
# 694
constexpr bool isgreater(double __x, double __y) 
# 695
{ return __builtin_isgreater(__x, __y); } 
# 698
constexpr bool isgreater(long double __x, long double __y) 
# 699
{ return __builtin_isgreater(__x, __y); } 
# 703
template< class _Tp, class _Up> constexpr typename __gnu_cxx::__enable_if< __is_arithmetic< _Tp> ::__value && __is_arithmetic< _Up> ::__value, bool> ::__type 
# 707
isgreater(_Tp __x, _Up __y) 
# 708
{ 
# 709
typedef typename __gnu_cxx::__promote_2< _Tp, _Up> ::__type __type; 
# 710
return __builtin_isgreater((__type)__x, (__type)__y); 
# 711
} 
# 716
constexpr bool isgreaterequal(float __x, float __y) 
# 717
{ return __builtin_isgreaterequal(__x, __y); } 
# 720
constexpr bool isgreaterequal(double __x, double __y) 
# 721
{ return __builtin_isgreaterequal(__x, __y); } 
# 724
constexpr bool isgreaterequal(long double __x, long double __y) 
# 725
{ return __builtin_isgreaterequal(__x, __y); } 
# 729
template< class _Tp, class _Up> constexpr typename __gnu_cxx::__enable_if< __is_arithmetic< _Tp> ::__value && __is_arithmetic< _Up> ::__value, bool> ::__type 
# 733
isgreaterequal(_Tp __x, _Up __y) 
# 734
{ 
# 735
typedef typename __gnu_cxx::__promote_2< _Tp, _Up> ::__type __type; 
# 736
return __builtin_isgreaterequal((__type)__x, (__type)__y); 
# 737
} 
# 742
constexpr bool isless(float __x, float __y) 
# 743
{ return __builtin_isless(__x, __y); } 
# 746
constexpr bool isless(double __x, double __y) 
# 747
{ return __builtin_isless(__x, __y); } 
# 750
constexpr bool isless(long double __x, long double __y) 
# 751
{ return __builtin_isless(__x, __y); } 
# 755
template< class _Tp, class _Up> constexpr typename __gnu_cxx::__enable_if< __is_arithmetic< _Tp> ::__value && __is_arithmetic< _Up> ::__value, bool> ::__type 
# 759
isless(_Tp __x, _Up __y) 
# 760
{ 
# 761
typedef typename __gnu_cxx::__promote_2< _Tp, _Up> ::__type __type; 
# 762
return __builtin_isless((__type)__x, (__type)__y); 
# 763
} 
# 768
constexpr bool islessequal(float __x, float __y) 
# 769
{ return __builtin_islessequal(__x, __y); } 
# 772
constexpr bool islessequal(double __x, double __y) 
# 773
{ return __builtin_islessequal(__x, __y); } 
# 776
constexpr bool islessequal(long double __x, long double __y) 
# 777
{ return __builtin_islessequal(__x, __y); } 
# 781
template< class _Tp, class _Up> constexpr typename __gnu_cxx::__enable_if< __is_arithmetic< _Tp> ::__value && __is_arithmetic< _Up> ::__value, bool> ::__type 
# 785
islessequal(_Tp __x, _Up __y) 
# 786
{ 
# 787
typedef typename __gnu_cxx::__promote_2< _Tp, _Up> ::__type __type; 
# 788
return __builtin_islessequal((__type)__x, (__type)__y); 
# 789
} 
# 794
constexpr bool islessgreater(float __x, float __y) 
# 795
{ return __builtin_islessgreater(__x, __y); } 
# 798
constexpr bool islessgreater(double __x, double __y) 
# 799
{ return __builtin_islessgreater(__x, __y); } 
# 802
constexpr bool islessgreater(long double __x, long double __y) 
# 803
{ return __builtin_islessgreater(__x, __y); } 
# 807
template< class _Tp, class _Up> constexpr typename __gnu_cxx::__enable_if< __is_arithmetic< _Tp> ::__value && __is_arithmetic< _Up> ::__value, bool> ::__type 
# 811
islessgreater(_Tp __x, _Up __y) 
# 812
{ 
# 813
typedef typename __gnu_cxx::__promote_2< _Tp, _Up> ::__type __type; 
# 814
return __builtin_islessgreater((__type)__x, (__type)__y); 
# 815
} 
# 820
constexpr bool isunordered(float __x, float __y) 
# 821
{ return __builtin_isunordered(__x, __y); } 
# 824
constexpr bool isunordered(double __x, double __y) 
# 825
{ return __builtin_isunordered(__x, __y); } 
# 828
constexpr bool isunordered(long double __x, long double __y) 
# 829
{ return __builtin_isunordered(__x, __y); } 
# 833
template< class _Tp, class _Up> constexpr typename __gnu_cxx::__enable_if< __is_arithmetic< _Tp> ::__value && __is_arithmetic< _Up> ::__value, bool> ::__type 
# 837
isunordered(_Tp __x, _Up __y) 
# 838
{ 
# 839
typedef typename __gnu_cxx::__promote_2< _Tp, _Up> ::__type __type; 
# 840
return __builtin_isunordered((__type)__x, (__type)__y); 
# 841
} 
# 958 "/opt/rh/devtoolset-7/root/usr/include/c++/7/cmath" 3
}
# 1075 "/opt/rh/devtoolset-7/root/usr/include/c++/7/cmath" 3
namespace std __attribute((__visibility__("default"))) { 
# 1080
using ::double_t;
# 1081
using ::float_t;
# 1084
using ::acosh;
# 1085
using ::acoshf;
# 1086
using ::acoshl;
# 1088
using ::asinh;
# 1089
using ::asinhf;
# 1090
using ::asinhl;
# 1092
using ::atanh;
# 1093
using ::atanhf;
# 1094
using ::atanhl;
# 1096
using ::cbrt;
# 1097
using ::cbrtf;
# 1098
using ::cbrtl;
# 1100
using ::copysign;
# 1101
using ::copysignf;
# 1102
using ::copysignl;
# 1104
using ::erf;
# 1105
using ::erff;
# 1106
using ::erfl;
# 1108
using ::erfc;
# 1109
using ::erfcf;
# 1110
using ::erfcl;
# 1112
using ::exp2;
# 1113
using ::exp2f;
# 1114
using ::exp2l;
# 1116
using ::expm1;
# 1117
using ::expm1f;
# 1118
using ::expm1l;
# 1120
using ::fdim;
# 1121
using ::fdimf;
# 1122
using ::fdiml;
# 1124
using ::fma;
# 1125
using ::fmaf;
# 1126
using ::fmal;
# 1128
using ::fmax;
# 1129
using ::fmaxf;
# 1130
using ::fmaxl;
# 1132
using ::fmin;
# 1133
using ::fminf;
# 1134
using ::fminl;
# 1136
using ::hypot;
# 1137
using ::hypotf;
# 1138
using ::hypotl;
# 1140
using ::ilogb;
# 1141
using ::ilogbf;
# 1142
using ::ilogbl;
# 1144
using ::lgamma;
# 1145
using ::lgammaf;
# 1146
using ::lgammal;
# 1149
using ::llrint;
# 1150
using ::llrintf;
# 1151
using ::llrintl;
# 1153
using ::llround;
# 1154
using ::llroundf;
# 1155
using ::llroundl;
# 1158
using ::log1p;
# 1159
using ::log1pf;
# 1160
using ::log1pl;
# 1162
using ::log2;
# 1163
using ::log2f;
# 1164
using ::log2l;
# 1166
using ::logb;
# 1167
using ::logbf;
# 1168
using ::logbl;
# 1170
using ::lrint;
# 1171
using ::lrintf;
# 1172
using ::lrintl;
# 1174
using ::lround;
# 1175
using ::lroundf;
# 1176
using ::lroundl;
# 1178
using ::nan;
# 1179
using ::nanf;
# 1180
using ::nanl;
# 1182
using ::nearbyint;
# 1183
using ::nearbyintf;
# 1184
using ::nearbyintl;
# 1186
using ::nextafter;
# 1187
using ::nextafterf;
# 1188
using ::nextafterl;
# 1190
using ::nexttoward;
# 1191
using ::nexttowardf;
# 1192
using ::nexttowardl;
# 1194
using ::remainder;
# 1195
using ::remainderf;
# 1196
using ::remainderl;
# 1198
using ::remquo;
# 1199
using ::remquof;
# 1200
using ::remquol;
# 1202
using ::rint;
# 1203
using ::rintf;
# 1204
using ::rintl;
# 1206
using ::round;
# 1207
using ::roundf;
# 1208
using ::roundl;
# 1210
using ::scalbln;
# 1211
using ::scalblnf;
# 1212
using ::scalblnl;
# 1214
using ::scalbn;
# 1215
using ::scalbnf;
# 1216
using ::scalbnl;
# 1218
using ::tgamma;
# 1219
using ::tgammaf;
# 1220
using ::tgammal;
# 1222
using ::trunc;
# 1223
using ::truncf;
# 1224
using ::truncl;
# 1229
constexpr float acosh(float __x) 
# 1230
{ return __builtin_acoshf(__x); } 
# 1233
constexpr long double acosh(long double __x) 
# 1234
{ return __builtin_acoshl(__x); } 
# 1238
template< class _Tp> constexpr typename __gnu_cxx::__enable_if< __is_integer< _Tp> ::__value, double> ::__type 
# 1241
acosh(_Tp __x) 
# 1242
{ return __builtin_acosh(__x); } 
# 1247
constexpr float asinh(float __x) 
# 1248
{ return __builtin_asinhf(__x); } 
# 1251
constexpr long double asinh(long double __x) 
# 1252
{ return __builtin_asinhl(__x); } 
# 1256
template< class _Tp> constexpr typename __gnu_cxx::__enable_if< __is_integer< _Tp> ::__value, double> ::__type 
# 1259
asinh(_Tp __x) 
# 1260
{ return __builtin_asinh(__x); } 
# 1265
constexpr float atanh(float __x) 
# 1266
{ return __builtin_atanhf(__x); } 
# 1269
constexpr long double atanh(long double __x) 
# 1270
{ return __builtin_atanhl(__x); } 
# 1274
template< class _Tp> constexpr typename __gnu_cxx::__enable_if< __is_integer< _Tp> ::__value, double> ::__type 
# 1277
atanh(_Tp __x) 
# 1278
{ return __builtin_atanh(__x); } 
# 1283
constexpr float cbrt(float __x) 
# 1284
{ return __builtin_cbrtf(__x); } 
# 1287
constexpr long double cbrt(long double __x) 
# 1288
{ return __builtin_cbrtl(__x); } 
# 1292
template< class _Tp> constexpr typename __gnu_cxx::__enable_if< __is_integer< _Tp> ::__value, double> ::__type 
# 1295
cbrt(_Tp __x) 
# 1296
{ return __builtin_cbrt(__x); } 
# 1301
constexpr float copysign(float __x, float __y) 
# 1302
{ return __builtin_copysignf(__x, __y); } 
# 1305
constexpr long double copysign(long double __x, long double __y) 
# 1306
{ return __builtin_copysignl(__x, __y); } 
# 1310
template< class _Tp, class _Up> constexpr typename __gnu_cxx::__promote_2< _Tp, _Up> ::__type 
# 1312
copysign(_Tp __x, _Up __y) 
# 1313
{ 
# 1314
typedef typename __gnu_cxx::__promote_2< _Tp, _Up> ::__type __type; 
# 1315
return copysign((__type)__x, (__type)__y); 
# 1316
} 
# 1321
constexpr float erf(float __x) 
# 1322
{ return __builtin_erff(__x); } 
# 1325
constexpr long double erf(long double __x) 
# 1326
{ return __builtin_erfl(__x); } 
# 1330
template< class _Tp> constexpr typename __gnu_cxx::__enable_if< __is_integer< _Tp> ::__value, double> ::__type 
# 1333
erf(_Tp __x) 
# 1334
{ return __builtin_erf(__x); } 
# 1339
constexpr float erfc(float __x) 
# 1340
{ return __builtin_erfcf(__x); } 
# 1343
constexpr long double erfc(long double __x) 
# 1344
{ return __builtin_erfcl(__x); } 
# 1348
template< class _Tp> constexpr typename __gnu_cxx::__enable_if< __is_integer< _Tp> ::__value, double> ::__type 
# 1351
erfc(_Tp __x) 
# 1352
{ return __builtin_erfc(__x); } 
# 1357
constexpr float exp2(float __x) 
# 1358
{ return __builtin_exp2f(__x); } 
# 1361
constexpr long double exp2(long double __x) 
# 1362
{ return __builtin_exp2l(__x); } 
# 1366
template< class _Tp> constexpr typename __gnu_cxx::__enable_if< __is_integer< _Tp> ::__value, double> ::__type 
# 1369
exp2(_Tp __x) 
# 1370
{ return __builtin_exp2(__x); } 
# 1375
constexpr float expm1(float __x) 
# 1376
{ return __builtin_expm1f(__x); } 
# 1379
constexpr long double expm1(long double __x) 
# 1380
{ return __builtin_expm1l(__x); } 
# 1384
template< class _Tp> constexpr typename __gnu_cxx::__enable_if< __is_integer< _Tp> ::__value, double> ::__type 
# 1387
expm1(_Tp __x) 
# 1388
{ return __builtin_expm1(__x); } 
# 1393
constexpr float fdim(float __x, float __y) 
# 1394
{ return __builtin_fdimf(__x, __y); } 
# 1397
constexpr long double fdim(long double __x, long double __y) 
# 1398
{ return __builtin_fdiml(__x, __y); } 
# 1402
template< class _Tp, class _Up> constexpr typename __gnu_cxx::__promote_2< _Tp, _Up> ::__type 
# 1404
fdim(_Tp __x, _Up __y) 
# 1405
{ 
# 1406
typedef typename __gnu_cxx::__promote_2< _Tp, _Up> ::__type __type; 
# 1407
return fdim((__type)__x, (__type)__y); 
# 1408
} 
# 1413
constexpr float fma(float __x, float __y, float __z) 
# 1414
{ return __builtin_fmaf(__x, __y, __z); } 
# 1417
constexpr long double fma(long double __x, long double __y, long double __z) 
# 1418
{ return __builtin_fmal(__x, __y, __z); } 
# 1422
template< class _Tp, class _Up, class _Vp> constexpr typename __gnu_cxx::__promote_3< _Tp, _Up, _Vp> ::__type 
# 1424
fma(_Tp __x, _Up __y, _Vp __z) 
# 1425
{ 
# 1426
typedef typename __gnu_cxx::__promote_3< _Tp, _Up, _Vp> ::__type __type; 
# 1427
return fma((__type)__x, (__type)__y, (__type)__z); 
# 1428
} 
# 1433
constexpr float fmax(float __x, float __y) 
# 1434
{ return __builtin_fmaxf(__x, __y); } 
# 1437
constexpr long double fmax(long double __x, long double __y) 
# 1438
{ return __builtin_fmaxl(__x, __y); } 
# 1442
template< class _Tp, class _Up> constexpr typename __gnu_cxx::__promote_2< _Tp, _Up> ::__type 
# 1444
fmax(_Tp __x, _Up __y) 
# 1445
{ 
# 1446
typedef typename __gnu_cxx::__promote_2< _Tp, _Up> ::__type __type; 
# 1447
return fmax((__type)__x, (__type)__y); 
# 1448
} 
# 1453
constexpr float fmin(float __x, float __y) 
# 1454
{ return __builtin_fminf(__x, __y); } 
# 1457
constexpr long double fmin(long double __x, long double __y) 
# 1458
{ return __builtin_fminl(__x, __y); } 
# 1462
template< class _Tp, class _Up> constexpr typename __gnu_cxx::__promote_2< _Tp, _Up> ::__type 
# 1464
fmin(_Tp __x, _Up __y) 
# 1465
{ 
# 1466
typedef typename __gnu_cxx::__promote_2< _Tp, _Up> ::__type __type; 
# 1467
return fmin((__type)__x, (__type)__y); 
# 1468
} 
# 1473
constexpr float hypot(float __x, float __y) 
# 1474
{ return __builtin_hypotf(__x, __y); } 
# 1477
constexpr long double hypot(long double __x, long double __y) 
# 1478
{ return __builtin_hypotl(__x, __y); } 
# 1482
template< class _Tp, class _Up> constexpr typename __gnu_cxx::__promote_2< _Tp, _Up> ::__type 
# 1484
hypot(_Tp __x, _Up __y) 
# 1485
{ 
# 1486
typedef typename __gnu_cxx::__promote_2< _Tp, _Up> ::__type __type; 
# 1487
return hypot((__type)__x, (__type)__y); 
# 1488
} 
# 1493
constexpr int ilogb(float __x) 
# 1494
{ return __builtin_ilogbf(__x); } 
# 1497
constexpr int ilogb(long double __x) 
# 1498
{ return __builtin_ilogbl(__x); } 
# 1502
template< class _Tp> constexpr typename __gnu_cxx::__enable_if< __is_integer< _Tp> ::__value, int> ::__type 
# 1506
ilogb(_Tp __x) 
# 1507
{ return __builtin_ilogb(__x); } 
# 1512
constexpr float lgamma(float __x) 
# 1513
{ return __builtin_lgammaf(__x); } 
# 1516
constexpr long double lgamma(long double __x) 
# 1517
{ return __builtin_lgammal(__x); } 
# 1521
template< class _Tp> constexpr typename __gnu_cxx::__enable_if< __is_integer< _Tp> ::__value, double> ::__type 
# 1524
lgamma(_Tp __x) 
# 1525
{ return __builtin_lgamma(__x); } 
# 1530
constexpr long long llrint(float __x) 
# 1531
{ return __builtin_llrintf(__x); } 
# 1534
constexpr long long llrint(long double __x) 
# 1535
{ return __builtin_llrintl(__x); } 
# 1539
template< class _Tp> constexpr typename __gnu_cxx::__enable_if< __is_integer< _Tp> ::__value, long long> ::__type 
# 1542
llrint(_Tp __x) 
# 1543
{ return __builtin_llrint(__x); } 
# 1548
constexpr long long llround(float __x) 
# 1549
{ return __builtin_llroundf(__x); } 
# 1552
constexpr long long llround(long double __x) 
# 1553
{ return __builtin_llroundl(__x); } 
# 1557
template< class _Tp> constexpr typename __gnu_cxx::__enable_if< __is_integer< _Tp> ::__value, long long> ::__type 
# 1560
llround(_Tp __x) 
# 1561
{ return __builtin_llround(__x); } 
# 1566
constexpr float log1p(float __x) 
# 1567
{ return __builtin_log1pf(__x); } 
# 1570
constexpr long double log1p(long double __x) 
# 1571
{ return __builtin_log1pl(__x); } 
# 1575
template< class _Tp> constexpr typename __gnu_cxx::__enable_if< __is_integer< _Tp> ::__value, double> ::__type 
# 1578
log1p(_Tp __x) 
# 1579
{ return __builtin_log1p(__x); } 
# 1585
constexpr float log2(float __x) 
# 1586
{ return __builtin_log2f(__x); } 
# 1589
constexpr long double log2(long double __x) 
# 1590
{ return __builtin_log2l(__x); } 
# 1594
template< class _Tp> constexpr typename __gnu_cxx::__enable_if< __is_integer< _Tp> ::__value, double> ::__type 
# 1597
log2(_Tp __x) 
# 1598
{ return __builtin_log2(__x); } 
# 1603
constexpr float logb(float __x) 
# 1604
{ return __builtin_logbf(__x); } 
# 1607
constexpr long double logb(long double __x) 
# 1608
{ return __builtin_logbl(__x); } 
# 1612
template< class _Tp> constexpr typename __gnu_cxx::__enable_if< __is_integer< _Tp> ::__value, double> ::__type 
# 1615
logb(_Tp __x) 
# 1616
{ return __builtin_logb(__x); } 
# 1621
constexpr long lrint(float __x) 
# 1622
{ return __builtin_lrintf(__x); } 
# 1625
constexpr long lrint(long double __x) 
# 1626
{ return __builtin_lrintl(__x); } 
# 1630
template< class _Tp> constexpr typename __gnu_cxx::__enable_if< __is_integer< _Tp> ::__value, long> ::__type 
# 1633
lrint(_Tp __x) 
# 1634
{ return __builtin_lrint(__x); } 
# 1639
constexpr long lround(float __x) 
# 1640
{ return __builtin_lroundf(__x); } 
# 1643
constexpr long lround(long double __x) 
# 1644
{ return __builtin_lroundl(__x); } 
# 1648
template< class _Tp> constexpr typename __gnu_cxx::__enable_if< __is_integer< _Tp> ::__value, long> ::__type 
# 1651
lround(_Tp __x) 
# 1652
{ return __builtin_lround(__x); } 
# 1657
constexpr float nearbyint(float __x) 
# 1658
{ return __builtin_nearbyintf(__x); } 
# 1661
constexpr long double nearbyint(long double __x) 
# 1662
{ return __builtin_nearbyintl(__x); } 
# 1666
template< class _Tp> constexpr typename __gnu_cxx::__enable_if< __is_integer< _Tp> ::__value, double> ::__type 
# 1669
nearbyint(_Tp __x) 
# 1670
{ return __builtin_nearbyint(__x); } 
# 1675
constexpr float nextafter(float __x, float __y) 
# 1676
{ return __builtin_nextafterf(__x, __y); } 
# 1679
constexpr long double nextafter(long double __x, long double __y) 
# 1680
{ return __builtin_nextafterl(__x, __y); } 
# 1684
template< class _Tp, class _Up> constexpr typename __gnu_cxx::__promote_2< _Tp, _Up> ::__type 
# 1686
nextafter(_Tp __x, _Up __y) 
# 1687
{ 
# 1688
typedef typename __gnu_cxx::__promote_2< _Tp, _Up> ::__type __type; 
# 1689
return nextafter((__type)__x, (__type)__y); 
# 1690
} 
# 1695
constexpr float nexttoward(float __x, long double __y) 
# 1696
{ return __builtin_nexttowardf(__x, __y); } 
# 1699
constexpr long double nexttoward(long double __x, long double __y) 
# 1700
{ return __builtin_nexttowardl(__x, __y); } 
# 1704
template< class _Tp> constexpr typename __gnu_cxx::__enable_if< __is_integer< _Tp> ::__value, double> ::__type 
# 1707
nexttoward(_Tp __x, long double __y) 
# 1708
{ return __builtin_nexttoward(__x, __y); } 
# 1713
constexpr float remainder(float __x, float __y) 
# 1714
{ return __builtin_remainderf(__x, __y); } 
# 1717
constexpr long double remainder(long double __x, long double __y) 
# 1718
{ return __builtin_remainderl(__x, __y); } 
# 1722
template< class _Tp, class _Up> constexpr typename __gnu_cxx::__promote_2< _Tp, _Up> ::__type 
# 1724
remainder(_Tp __x, _Up __y) 
# 1725
{ 
# 1726
typedef typename __gnu_cxx::__promote_2< _Tp, _Up> ::__type __type; 
# 1727
return remainder((__type)__x, (__type)__y); 
# 1728
} 
# 1733
inline float remquo(float __x, float __y, int *__pquo) 
# 1734
{ return __builtin_remquof(__x, __y, __pquo); } 
# 1737
inline long double remquo(long double __x, long double __y, int *__pquo) 
# 1738
{ return __builtin_remquol(__x, __y, __pquo); } 
# 1742
template< class _Tp, class _Up> inline typename __gnu_cxx::__promote_2< _Tp, _Up> ::__type 
# 1744
remquo(_Tp __x, _Up __y, int *__pquo) 
# 1745
{ 
# 1746
typedef typename __gnu_cxx::__promote_2< _Tp, _Up> ::__type __type; 
# 1747
return remquo((__type)__x, (__type)__y, __pquo); 
# 1748
} 
# 1753
constexpr float rint(float __x) 
# 1754
{ return __builtin_rintf(__x); } 
# 1757
constexpr long double rint(long double __x) 
# 1758
{ return __builtin_rintl(__x); } 
# 1762
template< class _Tp> constexpr typename __gnu_cxx::__enable_if< __is_integer< _Tp> ::__value, double> ::__type 
# 1765
rint(_Tp __x) 
# 1766
{ return __builtin_rint(__x); } 
# 1771
constexpr float round(float __x) 
# 1772
{ return __builtin_roundf(__x); } 
# 1775
constexpr long double round(long double __x) 
# 1776
{ return __builtin_roundl(__x); } 
# 1780
template< class _Tp> constexpr typename __gnu_cxx::__enable_if< __is_integer< _Tp> ::__value, double> ::__type 
# 1783
round(_Tp __x) 
# 1784
{ return __builtin_round(__x); } 
# 1789
constexpr float scalbln(float __x, long __ex) 
# 1790
{ return __builtin_scalblnf(__x, __ex); } 
# 1793
constexpr long double scalbln(long double __x, long __ex) 
# 1794
{ return __builtin_scalblnl(__x, __ex); } 
# 1798
template< class _Tp> constexpr typename __gnu_cxx::__enable_if< __is_integer< _Tp> ::__value, double> ::__type 
# 1801
scalbln(_Tp __x, long __ex) 
# 1802
{ return __builtin_scalbln(__x, __ex); } 
# 1807
constexpr float scalbn(float __x, int __ex) 
# 1808
{ return __builtin_scalbnf(__x, __ex); } 
# 1811
constexpr long double scalbn(long double __x, int __ex) 
# 1812
{ return __builtin_scalbnl(__x, __ex); } 
# 1816
template< class _Tp> constexpr typename __gnu_cxx::__enable_if< __is_integer< _Tp> ::__value, double> ::__type 
# 1819
scalbn(_Tp __x, int __ex) 
# 1820
{ return __builtin_scalbn(__x, __ex); } 
# 1825
constexpr float tgamma(float __x) 
# 1826
{ return __builtin_tgammaf(__x); } 
# 1829
constexpr long double tgamma(long double __x) 
# 1830
{ return __builtin_tgammal(__x); } 
# 1834
template< class _Tp> constexpr typename __gnu_cxx::__enable_if< __is_integer< _Tp> ::__value, double> ::__type 
# 1837
tgamma(_Tp __x) 
# 1838
{ return __builtin_tgamma(__x); } 
# 1843
constexpr float trunc(float __x) 
# 1844
{ return __builtin_truncf(__x); } 
# 1847
constexpr long double trunc(long double __x) 
# 1848
{ return __builtin_truncl(__x); } 
# 1852
template< class _Tp> constexpr typename __gnu_cxx::__enable_if< __is_integer< _Tp> ::__value, double> ::__type 
# 1855
trunc(_Tp __x) 
# 1856
{ return __builtin_trunc(__x); } 
# 1860
}
# 1917 "/opt/rh/devtoolset-7/root/usr/include/c++/7/cmath" 3
}
# 38 "/opt/rh/devtoolset-7/root/usr/include/c++/7/math.h" 3
using std::abs;
# 39
using std::acos;
# 40
using std::asin;
# 41
using std::atan;
# 42
using std::atan2;
# 43
using std::cos;
# 44
using std::sin;
# 45
using std::tan;
# 46
using std::cosh;
# 47
using std::sinh;
# 48
using std::tanh;
# 49
using std::exp;
# 50
using std::frexp;
# 51
using std::ldexp;
# 52
using std::log;
# 53
using std::log10;
# 54
using std::modf;
# 55
using std::pow;
# 56
using std::sqrt;
# 57
using std::ceil;
# 58
using std::fabs;
# 59
using std::floor;
# 60
using std::fmod;
# 63
using std::fpclassify;
# 64
using std::isfinite;
# 65
using std::isinf;
# 66
using std::isnan;
# 67
using std::isnormal;
# 68
using std::signbit;
# 69
using std::isgreater;
# 70
using std::isgreaterequal;
# 71
using std::isless;
# 72
using std::islessequal;
# 73
using std::islessgreater;
# 74
using std::isunordered;
# 78
using std::acosh;
# 79
using std::asinh;
# 80
using std::atanh;
# 81
using std::cbrt;
# 82
using std::copysign;
# 83
using std::erf;
# 84
using std::erfc;
# 85
using std::exp2;
# 86
using std::expm1;
# 87
using std::fdim;
# 88
using std::fma;
# 89
using std::fmax;
# 90
using std::fmin;
# 91
using std::hypot;
# 92
using std::ilogb;
# 93
using std::lgamma;
# 94
using std::llrint;
# 95
using std::llround;
# 96
using std::log1p;
# 97
using std::log2;
# 98
using std::logb;
# 99
using std::lrint;
# 100
using std::lround;
# 101
using std::nearbyint;
# 102
using std::nextafter;
# 103
using std::nexttoward;
# 104
using std::remainder;
# 105
using std::remquo;
# 106
using std::rint;
# 107
using std::round;
# 108
using std::scalbln;
# 109
using std::scalbn;
# 110
using std::tgamma;
# 111
using std::trunc;
# 118 "/opt/rh/devtoolset-7/root/usr/include/c++/7/cstdlib" 3
extern "C++" {
# 120
namespace std __attribute((__visibility__("default"))) { 
# 124
using ::div_t;
# 125
using ::ldiv_t;
# 127
using ::abort;
# 128
using ::atexit;
# 131
using ::at_quick_exit;
# 134
using ::atof;
# 135
using ::atoi;
# 136
using ::atol;
# 137
using ::bsearch;
# 138
using ::calloc;
# 139
using ::div;
# 140
using ::exit;
# 141
using ::free;
# 142
using ::getenv;
# 143
using ::labs;
# 144
using ::ldiv;
# 145
using ::malloc;
# 147
using ::mblen;
# 148
using ::mbstowcs;
# 149
using ::mbtowc;
# 151
using ::qsort;
# 154
using ::quick_exit;
# 157
using ::rand;
# 158
using ::realloc;
# 159
using ::srand;
# 160
using ::strtod;
# 161
using ::strtol;
# 162
using ::strtoul;
# 163
using ::system;
# 165
using ::wcstombs;
# 166
using ::wctomb;
# 171
inline ldiv_t div(long __i, long __j) { return ldiv(__i, __j); } 
# 176
}
# 189 "/opt/rh/devtoolset-7/root/usr/include/c++/7/cstdlib" 3
namespace __gnu_cxx __attribute((__visibility__("default"))) { 
# 194
using ::lldiv_t;
# 200
using ::_Exit;
# 204
using ::llabs;
# 207
inline lldiv_t div(long long __n, long long __d) 
# 208
{ lldiv_t __q; (__q.quot) = (__n / __d); (__q.rem) = (__n % __d); return __q; } 
# 210
using ::lldiv;
# 221 "/opt/rh/devtoolset-7/root/usr/include/c++/7/cstdlib" 3
using ::atoll;
# 222
using ::strtoll;
# 223
using ::strtoull;
# 225
using ::strtof;
# 226
using ::strtold;
# 229
}
# 231
namespace std { 
# 234
using __gnu_cxx::lldiv_t;
# 236
using __gnu_cxx::_Exit;
# 238
using __gnu_cxx::llabs;
# 239
using __gnu_cxx::div;
# 240
using __gnu_cxx::lldiv;
# 242
using __gnu_cxx::atoll;
# 243
using __gnu_cxx::strtof;
# 244
using __gnu_cxx::strtoll;
# 245
using __gnu_cxx::strtoull;
# 246
using __gnu_cxx::strtold;
# 247
}
# 251
}
# 38 "/opt/rh/devtoolset-7/root/usr/include/c++/7/stdlib.h" 3
using std::abort;
# 39
using std::atexit;
# 40
using std::exit;
# 43
using std::at_quick_exit;
# 46
using std::quick_exit;
# 54
using std::abs;
# 55
using std::atof;
# 56
using std::atoi;
# 57
using std::atol;
# 58
using std::bsearch;
# 59
using std::calloc;
# 60
using std::div;
# 61
using std::free;
# 62
using std::getenv;
# 63
using std::labs;
# 64
using std::ldiv;
# 65
using std::malloc;
# 67
using std::mblen;
# 68
using std::mbstowcs;
# 69
using std::mbtowc;
# 71
using std::qsort;
# 72
using std::rand;
# 73
using std::realloc;
# 74
using std::srand;
# 75
using std::strtod;
# 76
using std::strtol;
# 77
using std::strtoul;
# 78
using std::system;
# 80
using std::wcstombs;
# 81
using std::wctomb;
# 9029 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/crt/math_functions.h"
namespace std { 
# 9030
constexpr bool signbit(float x); 
# 9031
constexpr bool signbit(double x); 
# 9032
constexpr bool signbit(long double x); 
# 9033
constexpr bool isfinite(float x); 
# 9034
constexpr bool isfinite(double x); 
# 9035
constexpr bool isfinite(long double x); 
# 9036
constexpr bool isnan(float x); 
# 9039
extern "C" int isnan(double x) throw(); 
# 9043
constexpr bool isnan(long double x); 
# 9044
constexpr bool isinf(float x); 
# 9047
extern "C" int isinf(double x) throw(); 
# 9051
constexpr bool isinf(long double x); 
# 9052
}
# 9205 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/crt/math_functions.h"
namespace std { 
# 9207
template< class T> extern T __pow_helper(T, int); 
# 9208
template< class T> extern T __cmath_power(T, unsigned); 
# 9209
}
# 9211
using std::abs;
# 9212
using std::fabs;
# 9213
using std::ceil;
# 9214
using std::floor;
# 9215
using std::sqrt;
# 9217
using std::pow;
# 9219
using std::log;
# 9220
using std::log10;
# 9221
using std::fmod;
# 9222
using std::modf;
# 9223
using std::exp;
# 9224
using std::frexp;
# 9225
using std::ldexp;
# 9226
using std::asin;
# 9227
using std::sin;
# 9228
using std::sinh;
# 9229
using std::acos;
# 9230
using std::cos;
# 9231
using std::cosh;
# 9232
using std::atan;
# 9233
using std::atan2;
# 9234
using std::tan;
# 9235
using std::tanh;
# 9606 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/crt/math_functions.h"
namespace std { 
# 9615 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/crt/math_functions.h"
extern inline long long abs(long long); 
# 9625 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/crt/math_functions.h"
extern inline long abs(long); 
# 9626
extern constexpr float abs(float); 
# 9627
extern constexpr double abs(double); 
# 9628
extern constexpr float fabs(float); 
# 9629
extern constexpr float ceil(float); 
# 9630
extern constexpr float floor(float); 
# 9631
extern constexpr float sqrt(float); 
# 9632
extern constexpr float pow(float, float); 
# 9637
template< class _Tp, class _Up> extern constexpr typename __gnu_cxx::__promote_2< _Tp, _Up> ::__type pow(_Tp, _Up); 
# 9647
extern constexpr float log(float); 
# 9648
extern constexpr float log10(float); 
# 9649
extern constexpr float fmod(float, float); 
# 9650
extern inline float modf(float, float *); 
# 9651
extern constexpr float exp(float); 
# 9652
extern inline float frexp(float, int *); 
# 9653
extern constexpr float ldexp(float, int); 
# 9654
extern constexpr float asin(float); 
# 9655
extern constexpr float sin(float); 
# 9656
extern constexpr float sinh(float); 
# 9657
extern constexpr float acos(float); 
# 9658
extern constexpr float cos(float); 
# 9659
extern constexpr float cosh(float); 
# 9660
extern constexpr float atan(float); 
# 9661
extern constexpr float atan2(float, float); 
# 9662
extern constexpr float tan(float); 
# 9663
extern constexpr float tanh(float); 
# 9742 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/crt/math_functions.h"
}
# 9848 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/crt/math_functions.h"
namespace std { 
# 9849
constexpr float logb(float a); 
# 9850
constexpr int ilogb(float a); 
# 9851
constexpr float scalbn(float a, int b); 
# 9852
constexpr float scalbln(float a, long b); 
# 9853
constexpr float exp2(float a); 
# 9854
constexpr float expm1(float a); 
# 9855
constexpr float log2(float a); 
# 9856
constexpr float log1p(float a); 
# 9857
constexpr float acosh(float a); 
# 9858
constexpr float asinh(float a); 
# 9859
constexpr float atanh(float a); 
# 9860
constexpr float hypot(float a, float b); 
# 9861
constexpr float cbrt(float a); 
# 9862
constexpr float erf(float a); 
# 9863
constexpr float erfc(float a); 
# 9864
constexpr float lgamma(float a); 
# 9865
constexpr float tgamma(float a); 
# 9866
constexpr float copysign(float a, float b); 
# 9867
constexpr float nextafter(float a, float b); 
# 9868
constexpr float remainder(float a, float b); 
# 9869
inline float remquo(float a, float b, int * quo); 
# 9870
constexpr float round(float a); 
# 9871
constexpr long lround(float a); 
# 9872
constexpr long long llround(float a); 
# 9873
constexpr float trunc(float a); 
# 9874
constexpr float rint(float a); 
# 9875
constexpr long lrint(float a); 
# 9876
constexpr long long llrint(float a); 
# 9877
constexpr float nearbyint(float a); 
# 9878
constexpr float fdim(float a, float b); 
# 9879
constexpr float fma(float a, float b, float c); 
# 9880
constexpr float fmax(float a, float b); 
# 9881
constexpr float fmin(float a, float b); 
# 9882
}
# 9987 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/crt/math_functions.h"
static inline float exp10(const float a); 
# 9989
static inline float rsqrt(const float a); 
# 9991
static inline float rcbrt(const float a); 
# 9993
static inline float sinpi(const float a); 
# 9995
static inline float cospi(const float a); 
# 9997
static inline void sincospi(const float a, float *const sptr, float *const cptr); 
# 9999
static inline void sincos(const float a, float *const sptr, float *const cptr); 
# 10001
static inline float j0(const float a); 
# 10003
static inline float j1(const float a); 
# 10005
static inline float jn(const int n, const float a); 
# 10007
static inline float y0(const float a); 
# 10009
static inline float y1(const float a); 
# 10011
static inline float yn(const int n, const float a); 
# 10013
__attribute__((unused)) static inline float cyl_bessel_i0(const float a); 
# 10015
__attribute__((unused)) static inline float cyl_bessel_i1(const float a); 
# 10017
static inline float erfinv(const float a); 
# 10019
static inline float erfcinv(const float a); 
# 10021
static inline float normcdfinv(const float a); 
# 10023
static inline float normcdf(const float a); 
# 10025
static inline float erfcx(const float a); 
# 10027
static inline double copysign(const double a, const float b); 
# 10029
static inline double copysign(const float a, const double b); 
# 10037
static inline unsigned min(const unsigned a, const unsigned b); 
# 10045
static inline unsigned min(const int a, const unsigned b); 
# 10053
static inline unsigned min(const unsigned a, const int b); 
# 10061
static inline long min(const long a, const long b); 
# 10069
static inline unsigned long min(const unsigned long a, const unsigned long b); 
# 10077
static inline unsigned long min(const long a, const unsigned long b); 
# 10085
static inline unsigned long min(const unsigned long a, const long b); 
# 10093
static inline long long min(const long long a, const long long b); 
# 10101
static inline unsigned long long min(const unsigned long long a, const unsigned long long b); 
# 10109
static inline unsigned long long min(const long long a, const unsigned long long b); 
# 10117
static inline unsigned long long min(const unsigned long long a, const long long b); 
# 10128 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/crt/math_functions.h"
static inline float min(const float a, const float b); 
# 10139 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/crt/math_functions.h"
static inline double min(const double a, const double b); 
# 10149 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/crt/math_functions.h"
static inline double min(const float a, const double b); 
# 10159 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/crt/math_functions.h"
static inline double min(const double a, const float b); 
# 10167
static inline unsigned max(const unsigned a, const unsigned b); 
# 10175
static inline unsigned max(const int a, const unsigned b); 
# 10183
static inline unsigned max(const unsigned a, const int b); 
# 10191
static inline long max(const long a, const long b); 
# 10199
static inline unsigned long max(const unsigned long a, const unsigned long b); 
# 10207
static inline unsigned long max(const long a, const unsigned long b); 
# 10215
static inline unsigned long max(const unsigned long a, const long b); 
# 10223
static inline long long max(const long long a, const long long b); 
# 10231
static inline unsigned long long max(const unsigned long long a, const unsigned long long b); 
# 10239
static inline unsigned long long max(const long long a, const unsigned long long b); 
# 10247
static inline unsigned long long max(const unsigned long long a, const long long b); 
# 10258 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/crt/math_functions.h"
static inline float max(const float a, const float b); 
# 10269 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/crt/math_functions.h"
static inline double max(const double a, const double b); 
# 10279 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/crt/math_functions.h"
static inline double max(const float a, const double b); 
# 10289 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/crt/math_functions.h"
static inline double max(const double a, const float b); 
# 10300 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/crt/math_functions.h"
extern "C" {
# 10301
__attribute__((unused)) inline void *__nv_aligned_device_malloc(size_t size, size_t align) 
# 10302
{int volatile ___ = 1;(void)size;(void)align;
# 10305
::exit(___);}
#if 0
# 10302
{ 
# 10303
__attribute__((unused)) void *__nv_aligned_device_malloc_impl(size_t, size_t); 
# 10304
return __nv_aligned_device_malloc_impl(size, align); 
# 10305
} 
#endif
# 10306 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/crt/math_functions.h"
}
# 758 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/crt/math_functions.hpp"
static inline float exp10(const float a) 
# 759
{ 
# 760
return exp10f(a); 
# 761
} 
# 763
static inline float rsqrt(const float a) 
# 764
{ 
# 765
return rsqrtf(a); 
# 766
} 
# 768
static inline float rcbrt(const float a) 
# 769
{ 
# 770
return rcbrtf(a); 
# 771
} 
# 773
static inline float sinpi(const float a) 
# 774
{ 
# 775
return sinpif(a); 
# 776
} 
# 778
static inline float cospi(const float a) 
# 779
{ 
# 780
return cospif(a); 
# 781
} 
# 783
static inline void sincospi(const float a, float *const sptr, float *const cptr) 
# 784
{ 
# 785
sincospif(a, sptr, cptr); 
# 786
} 
# 788
static inline void sincos(const float a, float *const sptr, float *const cptr) 
# 789
{ 
# 790
sincosf(a, sptr, cptr); 
# 791
} 
# 793
static inline float j0(const float a) 
# 794
{ 
# 795
return j0f(a); 
# 796
} 
# 798
static inline float j1(const float a) 
# 799
{ 
# 800
return j1f(a); 
# 801
} 
# 803
static inline float jn(const int n, const float a) 
# 804
{ 
# 805
return jnf(n, a); 
# 806
} 
# 808
static inline float y0(const float a) 
# 809
{ 
# 810
return y0f(a); 
# 811
} 
# 813
static inline float y1(const float a) 
# 814
{ 
# 815
return y1f(a); 
# 816
} 
# 818
static inline float yn(const int n, const float a) 
# 819
{ 
# 820
return ynf(n, a); 
# 821
} 
# 823
__attribute__((unused)) static inline float cyl_bessel_i0(const float a) 
# 824
{int volatile ___ = 1;(void)a;
# 826
::exit(___);}
#if 0
# 824
{ 
# 825
return cyl_bessel_i0f(a); 
# 826
} 
#endif
# 828 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/crt/math_functions.hpp"
__attribute__((unused)) static inline float cyl_bessel_i1(const float a) 
# 829
{int volatile ___ = 1;(void)a;
# 831
::exit(___);}
#if 0
# 829
{ 
# 830
return cyl_bessel_i1f(a); 
# 831
} 
#endif
# 833 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/crt/math_functions.hpp"
static inline float erfinv(const float a) 
# 834
{ 
# 835
return erfinvf(a); 
# 836
} 
# 838
static inline float erfcinv(const float a) 
# 839
{ 
# 840
return erfcinvf(a); 
# 841
} 
# 843
static inline float normcdfinv(const float a) 
# 844
{ 
# 845
return normcdfinvf(a); 
# 846
} 
# 848
static inline float normcdf(const float a) 
# 849
{ 
# 850
return normcdff(a); 
# 851
} 
# 853
static inline float erfcx(const float a) 
# 854
{ 
# 855
return erfcxf(a); 
# 856
} 
# 858
static inline double copysign(const double a, const float b) 
# 859
{ 
# 860
return copysign(a, static_cast< double>(b)); 
# 861
} 
# 863
static inline double copysign(const float a, const double b) 
# 864
{ 
# 865
return copysign(static_cast< double>(a), b); 
# 866
} 
# 868
static inline unsigned min(const unsigned a, const unsigned b) 
# 869
{ 
# 870
return umin(a, b); 
# 871
} 
# 873
static inline unsigned min(const int a, const unsigned b) 
# 874
{ 
# 875
return umin(static_cast< unsigned>(a), b); 
# 876
} 
# 878
static inline unsigned min(const unsigned a, const int b) 
# 879
{ 
# 880
return umin(a, static_cast< unsigned>(b)); 
# 881
} 
# 883
static inline long min(const long a, const long b) 
# 884
{ 
# 885
long retval; 
# 891
if (sizeof(long) == sizeof(int)) { 
# 895
retval = (static_cast< long>(min(static_cast< int>(a), static_cast< int>(b)))); 
# 896
} else { 
# 897
retval = (static_cast< long>(llmin(static_cast< long long>(a), static_cast< long long>(b)))); 
# 898
}  
# 899
return retval; 
# 900
} 
# 902
static inline unsigned long min(const unsigned long a, const unsigned long b) 
# 903
{ 
# 904
unsigned long retval; 
# 908
if (sizeof(unsigned long) == sizeof(unsigned)) { 
# 912
retval = (static_cast< unsigned long>(umin(static_cast< unsigned>(a), static_cast< unsigned>(b)))); 
# 913
} else { 
# 914
retval = (static_cast< unsigned long>(ullmin(static_cast< unsigned long long>(a), static_cast< unsigned long long>(b)))); 
# 915
}  
# 916
return retval; 
# 917
} 
# 919
static inline unsigned long min(const long a, const unsigned long b) 
# 920
{ 
# 921
unsigned long retval; 
# 925
if (sizeof(unsigned long) == sizeof(unsigned)) { 
# 929
retval = (static_cast< unsigned long>(umin(static_cast< unsigned>(a), static_cast< unsigned>(b)))); 
# 930
} else { 
# 931
retval = (static_cast< unsigned long>(ullmin(static_cast< unsigned long long>(a), static_cast< unsigned long long>(b)))); 
# 932
}  
# 933
return retval; 
# 934
} 
# 936
static inline unsigned long min(const unsigned long a, const long b) 
# 937
{ 
# 938
unsigned long retval; 
# 942
if (sizeof(unsigned long) == sizeof(unsigned)) { 
# 946
retval = (static_cast< unsigned long>(umin(static_cast< unsigned>(a), static_cast< unsigned>(b)))); 
# 947
} else { 
# 948
retval = (static_cast< unsigned long>(ullmin(static_cast< unsigned long long>(a), static_cast< unsigned long long>(b)))); 
# 949
}  
# 950
return retval; 
# 951
} 
# 953
static inline long long min(const long long a, const long long b) 
# 954
{ 
# 955
return llmin(a, b); 
# 956
} 
# 958
static inline unsigned long long min(const unsigned long long a, const unsigned long long b) 
# 959
{ 
# 960
return ullmin(a, b); 
# 961
} 
# 963
static inline unsigned long long min(const long long a, const unsigned long long b) 
# 964
{ 
# 965
return ullmin(static_cast< unsigned long long>(a), b); 
# 966
} 
# 968
static inline unsigned long long min(const unsigned long long a, const long long b) 
# 969
{ 
# 970
return ullmin(a, static_cast< unsigned long long>(b)); 
# 971
} 
# 973
static inline float min(const float a, const float b) 
# 974
{ 
# 975
return fminf(a, b); 
# 976
} 
# 978
static inline double min(const double a, const double b) 
# 979
{ 
# 980
return fmin(a, b); 
# 981
} 
# 983
static inline double min(const float a, const double b) 
# 984
{ 
# 985
return fmin(static_cast< double>(a), b); 
# 986
} 
# 988
static inline double min(const double a, const float b) 
# 989
{ 
# 990
return fmin(a, static_cast< double>(b)); 
# 991
} 
# 993
static inline unsigned max(const unsigned a, const unsigned b) 
# 994
{ 
# 995
return umax(a, b); 
# 996
} 
# 998
static inline unsigned max(const int a, const unsigned b) 
# 999
{ 
# 1000
return umax(static_cast< unsigned>(a), b); 
# 1001
} 
# 1003
static inline unsigned max(const unsigned a, const int b) 
# 1004
{ 
# 1005
return umax(a, static_cast< unsigned>(b)); 
# 1006
} 
# 1008
static inline long max(const long a, const long b) 
# 1009
{ 
# 1010
long retval; 
# 1015
if (sizeof(long) == sizeof(int)) { 
# 1019
retval = (static_cast< long>(max(static_cast< int>(a), static_cast< int>(b)))); 
# 1020
} else { 
# 1021
retval = (static_cast< long>(llmax(static_cast< long long>(a), static_cast< long long>(b)))); 
# 1022
}  
# 1023
return retval; 
# 1024
} 
# 1026
static inline unsigned long max(const unsigned long a, const unsigned long b) 
# 1027
{ 
# 1028
unsigned long retval; 
# 1032
if (sizeof(unsigned long) == sizeof(unsigned)) { 
# 1036
retval = (static_cast< unsigned long>(umax(static_cast< unsigned>(a), static_cast< unsigned>(b)))); 
# 1037
} else { 
# 1038
retval = (static_cast< unsigned long>(ullmax(static_cast< unsigned long long>(a), static_cast< unsigned long long>(b)))); 
# 1039
}  
# 1040
return retval; 
# 1041
} 
# 1043
static inline unsigned long max(const long a, const unsigned long b) 
# 1044
{ 
# 1045
unsigned long retval; 
# 1049
if (sizeof(unsigned long) == sizeof(unsigned)) { 
# 1053
retval = (static_cast< unsigned long>(umax(static_cast< unsigned>(a), static_cast< unsigned>(b)))); 
# 1054
} else { 
# 1055
retval = (static_cast< unsigned long>(ullmax(static_cast< unsigned long long>(a), static_cast< unsigned long long>(b)))); 
# 1056
}  
# 1057
return retval; 
# 1058
} 
# 1060
static inline unsigned long max(const unsigned long a, const long b) 
# 1061
{ 
# 1062
unsigned long retval; 
# 1066
if (sizeof(unsigned long) == sizeof(unsigned)) { 
# 1070
retval = (static_cast< unsigned long>(umax(static_cast< unsigned>(a), static_cast< unsigned>(b)))); 
# 1071
} else { 
# 1072
retval = (static_cast< unsigned long>(ullmax(static_cast< unsigned long long>(a), static_cast< unsigned long long>(b)))); 
# 1073
}  
# 1074
return retval; 
# 1075
} 
# 1077
static inline long long max(const long long a, const long long b) 
# 1078
{ 
# 1079
return llmax(a, b); 
# 1080
} 
# 1082
static inline unsigned long long max(const unsigned long long a, const unsigned long long b) 
# 1083
{ 
# 1084
return ullmax(a, b); 
# 1085
} 
# 1087
static inline unsigned long long max(const long long a, const unsigned long long b) 
# 1088
{ 
# 1089
return ullmax(static_cast< unsigned long long>(a), b); 
# 1090
} 
# 1092
static inline unsigned long long max(const unsigned long long a, const long long b) 
# 1093
{ 
# 1094
return ullmax(a, static_cast< unsigned long long>(b)); 
# 1095
} 
# 1097
static inline float max(const float a, const float b) 
# 1098
{ 
# 1099
return fmaxf(a, b); 
# 1100
} 
# 1102
static inline double max(const double a, const double b) 
# 1103
{ 
# 1104
return fmax(a, b); 
# 1105
} 
# 1107
static inline double max(const float a, const double b) 
# 1108
{ 
# 1109
return fmax(static_cast< double>(a), b); 
# 1110
} 
# 1112
static inline double max(const double a, const float b) 
# 1113
{ 
# 1114
return fmax(a, static_cast< double>(b)); 
# 1115
} 
# 1126 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/crt/math_functions.hpp"
inline int min(const int a, const int b) 
# 1127
{ 
# 1128
return (a < b) ? a : b; 
# 1129
} 
# 1131
inline unsigned umin(const unsigned a, const unsigned b) 
# 1132
{ 
# 1133
return (a < b) ? a : b; 
# 1134
} 
# 1136
inline long long llmin(const long long a, const long long b) 
# 1137
{ 
# 1138
return (a < b) ? a : b; 
# 1139
} 
# 1141
inline unsigned long long ullmin(const unsigned long long a, const unsigned long long 
# 1142
b) 
# 1143
{ 
# 1144
return (a < b) ? a : b; 
# 1145
} 
# 1147
inline int max(const int a, const int b) 
# 1148
{ 
# 1149
return (a > b) ? a : b; 
# 1150
} 
# 1152
inline unsigned umax(const unsigned a, const unsigned b) 
# 1153
{ 
# 1154
return (a > b) ? a : b; 
# 1155
} 
# 1157
inline long long llmax(const long long a, const long long b) 
# 1158
{ 
# 1159
return (a > b) ? a : b; 
# 1160
} 
# 1162
inline unsigned long long ullmax(const unsigned long long a, const unsigned long long 
# 1163
b) 
# 1164
{ 
# 1165
return (a > b) ? a : b; 
# 1166
} 
# 74 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/cuda_surface_types.h"
template< class T, int dim = 1> 
# 75
struct surface : public surfaceReference { 
# 78
surface() 
# 79
{ 
# 80
(channelDesc) = cudaCreateChannelDesc< T> (); 
# 81
} 
# 83
surface(cudaChannelFormatDesc desc) 
# 84
{ 
# 85
(channelDesc) = desc; 
# 86
} 
# 88
}; 
# 90
template< int dim> 
# 91
struct surface< void, dim>  : public surfaceReference { 
# 94
surface() 
# 95
{ 
# 96
(channelDesc) = cudaCreateChannelDesc< void> (); 
# 97
} 
# 99
}; 
# 74 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/cuda_texture_types.h"
template< class T, int texType = 1, cudaTextureReadMode mode = cudaReadModeElementType> 
# 75
struct texture : public textureReference { 
# 78
texture(int norm = 0, cudaTextureFilterMode 
# 79
fMode = cudaFilterModePoint, cudaTextureAddressMode 
# 80
aMode = cudaAddressModeClamp) 
# 81
{ 
# 82
(normalized) = norm; 
# 83
(filterMode) = fMode; 
# 84
((addressMode)[0]) = aMode; 
# 85
((addressMode)[1]) = aMode; 
# 86
((addressMode)[2]) = aMode; 
# 87
(channelDesc) = cudaCreateChannelDesc< T> (); 
# 88
(sRGB) = 0; 
# 89
} 
# 91
texture(int norm, cudaTextureFilterMode 
# 92
fMode, cudaTextureAddressMode 
# 93
aMode, cudaChannelFormatDesc 
# 94
desc) 
# 95
{ 
# 96
(normalized) = norm; 
# 97
(filterMode) = fMode; 
# 98
((addressMode)[0]) = aMode; 
# 99
((addressMode)[1]) = aMode; 
# 100
((addressMode)[2]) = aMode; 
# 101
(channelDesc) = desc; 
# 102
(sRGB) = 0; 
# 103
} 
# 105
}; 
# 89 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/crt/device_functions.h"
extern "C" {
# 3215 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/crt/device_functions.h"
}
# 3223
__attribute__((unused)) static inline int mulhi(const int a, const int b); 
# 3225
__attribute__((unused)) static inline unsigned mulhi(const unsigned a, const unsigned b); 
# 3227
__attribute__((unused)) static inline unsigned mulhi(const int a, const unsigned b); 
# 3229
__attribute__((unused)) static inline unsigned mulhi(const unsigned a, const int b); 
# 3231
__attribute__((unused)) static inline long long mul64hi(const long long a, const long long b); 
# 3233
__attribute__((unused)) static inline unsigned long long mul64hi(const unsigned long long a, const unsigned long long b); 
# 3235
__attribute__((unused)) static inline unsigned long long mul64hi(const long long a, const unsigned long long b); 
# 3237
__attribute__((unused)) static inline unsigned long long mul64hi(const unsigned long long a, const long long b); 
# 3239
__attribute__((unused)) static inline int float_as_int(const float a); 
# 3241
__attribute__((unused)) static inline float int_as_float(const int a); 
# 3243
__attribute__((unused)) static inline unsigned float_as_uint(const float a); 
# 3245
__attribute__((unused)) static inline float uint_as_float(const unsigned a); 
# 3247
__attribute__((unused)) static inline float saturate(const float a); 
# 3249
__attribute__((unused)) static inline int mul24(const int a, const int b); 
# 3251
__attribute__((unused)) static inline unsigned umul24(const unsigned a, const unsigned b); 
# 3253
__attribute__((unused)) static inline int float2int(const float a, const cudaRoundMode mode = cudaRoundZero); 
# 3255
__attribute__((unused)) static inline unsigned float2uint(const float a, const cudaRoundMode mode = cudaRoundZero); 
# 3257
__attribute__((unused)) static inline float int2float(const int a, const cudaRoundMode mode = cudaRoundNearest); 
# 3259
__attribute__((unused)) static inline float uint2float(const unsigned a, const cudaRoundMode mode = cudaRoundNearest); 
# 90 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/crt/device_functions.hpp"
__attribute__((unused)) static inline int mulhi(const int a, const int b) 
# 91
{int volatile ___ = 1;(void)a;(void)b;
# 93
::exit(___);}
#if 0
# 91
{ 
# 92
return __mulhi(a, b); 
# 93
} 
#endif
# 95 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/crt/device_functions.hpp"
__attribute__((unused)) static inline unsigned mulhi(const unsigned a, const unsigned b) 
# 96
{int volatile ___ = 1;(void)a;(void)b;
# 98
::exit(___);}
#if 0
# 96
{ 
# 97
return __umulhi(a, b); 
# 98
} 
#endif
# 100 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/crt/device_functions.hpp"
__attribute__((unused)) static inline unsigned mulhi(const int a, const unsigned b) 
# 101
{int volatile ___ = 1;(void)a;(void)b;
# 103
::exit(___);}
#if 0
# 101
{ 
# 102
return __umulhi(static_cast< unsigned>(a), b); 
# 103
} 
#endif
# 105 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/crt/device_functions.hpp"
__attribute__((unused)) static inline unsigned mulhi(const unsigned a, const int b) 
# 106
{int volatile ___ = 1;(void)a;(void)b;
# 108
::exit(___);}
#if 0
# 106
{ 
# 107
return __umulhi(a, static_cast< unsigned>(b)); 
# 108
} 
#endif
# 110 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/crt/device_functions.hpp"
__attribute__((unused)) static inline long long mul64hi(const long long a, const long long b) 
# 111
{int volatile ___ = 1;(void)a;(void)b;
# 113
::exit(___);}
#if 0
# 111
{ 
# 112
return __mul64hi(a, b); 
# 113
} 
#endif
# 115 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/crt/device_functions.hpp"
__attribute__((unused)) static inline unsigned long long mul64hi(const unsigned long long a, const unsigned long long b) 
# 116
{int volatile ___ = 1;(void)a;(void)b;
# 118
::exit(___);}
#if 0
# 116
{ 
# 117
return __umul64hi(a, b); 
# 118
} 
#endif
# 120 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/crt/device_functions.hpp"
__attribute__((unused)) static inline unsigned long long mul64hi(const long long a, const unsigned long long b) 
# 121
{int volatile ___ = 1;(void)a;(void)b;
# 123
::exit(___);}
#if 0
# 121
{ 
# 122
return __umul64hi(static_cast< unsigned long long>(a), b); 
# 123
} 
#endif
# 125 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/crt/device_functions.hpp"
__attribute__((unused)) static inline unsigned long long mul64hi(const unsigned long long a, const long long b) 
# 126
{int volatile ___ = 1;(void)a;(void)b;
# 128
::exit(___);}
#if 0
# 126
{ 
# 127
return __umul64hi(a, static_cast< unsigned long long>(b)); 
# 128
} 
#endif
# 130 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/crt/device_functions.hpp"
__attribute__((unused)) static inline int float_as_int(const float a) 
# 131
{int volatile ___ = 1;(void)a;
# 133
::exit(___);}
#if 0
# 131
{ 
# 132
return __float_as_int(a); 
# 133
} 
#endif
# 135 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/crt/device_functions.hpp"
__attribute__((unused)) static inline float int_as_float(const int a) 
# 136
{int volatile ___ = 1;(void)a;
# 138
::exit(___);}
#if 0
# 136
{ 
# 137
return __int_as_float(a); 
# 138
} 
#endif
# 140 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/crt/device_functions.hpp"
__attribute__((unused)) static inline unsigned float_as_uint(const float a) 
# 141
{int volatile ___ = 1;(void)a;
# 143
::exit(___);}
#if 0
# 141
{ 
# 142
return __float_as_uint(a); 
# 143
} 
#endif
# 145 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/crt/device_functions.hpp"
__attribute__((unused)) static inline float uint_as_float(const unsigned a) 
# 146
{int volatile ___ = 1;(void)a;
# 148
::exit(___);}
#if 0
# 146
{ 
# 147
return __uint_as_float(a); 
# 148
} 
#endif
# 149 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/crt/device_functions.hpp"
__attribute__((unused)) static inline float saturate(const float a) 
# 150
{int volatile ___ = 1;(void)a;
# 152
::exit(___);}
#if 0
# 150
{ 
# 151
return __saturatef(a); 
# 152
} 
#endif
# 154 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/crt/device_functions.hpp"
__attribute__((unused)) static inline int mul24(const int a, const int b) 
# 155
{int volatile ___ = 1;(void)a;(void)b;
# 157
::exit(___);}
#if 0
# 155
{ 
# 156
return __mul24(a, b); 
# 157
} 
#endif
# 159 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/crt/device_functions.hpp"
__attribute__((unused)) static inline unsigned umul24(const unsigned a, const unsigned b) 
# 160
{int volatile ___ = 1;(void)a;(void)b;
# 162
::exit(___);}
#if 0
# 160
{ 
# 161
return __umul24(a, b); 
# 162
} 
#endif
# 164 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/crt/device_functions.hpp"
__attribute__((unused)) static inline int float2int(const float a, const cudaRoundMode mode) 
# 165
{int volatile ___ = 1;(void)a;(void)mode;
# 170
::exit(___);}
#if 0
# 165
{ 
# 166
return (mode == (cudaRoundNearest)) ? __float2int_rn(a) : ((mode == (cudaRoundPosInf)) ? __float2int_ru(a) : ((mode == (cudaRoundMinInf)) ? __float2int_rd(a) : __float2int_rz(a))); 
# 170
} 
#endif
# 172 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/crt/device_functions.hpp"
__attribute__((unused)) static inline unsigned float2uint(const float a, const cudaRoundMode mode) 
# 173
{int volatile ___ = 1;(void)a;(void)mode;
# 178
::exit(___);}
#if 0
# 173
{ 
# 174
return (mode == (cudaRoundNearest)) ? __float2uint_rn(a) : ((mode == (cudaRoundPosInf)) ? __float2uint_ru(a) : ((mode == (cudaRoundMinInf)) ? __float2uint_rd(a) : __float2uint_rz(a))); 
# 178
} 
#endif
# 180 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/crt/device_functions.hpp"
__attribute__((unused)) static inline float int2float(const int a, const cudaRoundMode mode) 
# 181
{int volatile ___ = 1;(void)a;(void)mode;
# 186
::exit(___);}
#if 0
# 181
{ 
# 182
return (mode == (cudaRoundZero)) ? __int2float_rz(a) : ((mode == (cudaRoundPosInf)) ? __int2float_ru(a) : ((mode == (cudaRoundMinInf)) ? __int2float_rd(a) : __int2float_rn(a))); 
# 186
} 
#endif
# 188 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/crt/device_functions.hpp"
__attribute__((unused)) static inline float uint2float(const unsigned a, const cudaRoundMode mode) 
# 189
{int volatile ___ = 1;(void)a;(void)mode;
# 194
::exit(___);}
#if 0
# 189
{ 
# 190
return (mode == (cudaRoundZero)) ? __uint2float_rz(a) : ((mode == (cudaRoundPosInf)) ? __uint2float_ru(a) : ((mode == (cudaRoundMinInf)) ? __uint2float_rd(a) : __uint2float_rn(a))); 
# 194
} 
#endif
# 113 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/device_atomic_functions.h"
__attribute__((unused)) static inline int atomicAdd(int *address, int val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
# 113
{ } 
#endif
# 115 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/device_atomic_functions.h"
__attribute__((unused)) static inline unsigned atomicAdd(unsigned *address, unsigned val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
# 115
{ } 
#endif
# 117 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/device_atomic_functions.h"
__attribute__((unused)) static inline int atomicSub(int *address, int val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
# 117
{ } 
#endif
# 119 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/device_atomic_functions.h"
__attribute__((unused)) static inline unsigned atomicSub(unsigned *address, unsigned val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
# 119
{ } 
#endif
# 121 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/device_atomic_functions.h"
__attribute__((unused)) static inline int atomicExch(int *address, int val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
# 121
{ } 
#endif
# 123 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/device_atomic_functions.h"
__attribute__((unused)) static inline unsigned atomicExch(unsigned *address, unsigned val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
# 123
{ } 
#endif
# 125 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/device_atomic_functions.h"
__attribute__((unused)) static inline float atomicExch(float *address, float val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
# 125
{ } 
#endif
# 127 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/device_atomic_functions.h"
__attribute__((unused)) static inline int atomicMin(int *address, int val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
# 127
{ } 
#endif
# 129 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/device_atomic_functions.h"
__attribute__((unused)) static inline unsigned atomicMin(unsigned *address, unsigned val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
# 129
{ } 
#endif
# 131 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/device_atomic_functions.h"
__attribute__((unused)) static inline int atomicMax(int *address, int val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
# 131
{ } 
#endif
# 133 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/device_atomic_functions.h"
__attribute__((unused)) static inline unsigned atomicMax(unsigned *address, unsigned val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
# 133
{ } 
#endif
# 135 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/device_atomic_functions.h"
__attribute__((unused)) static inline unsigned atomicInc(unsigned *address, unsigned val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
# 135
{ } 
#endif
# 137 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/device_atomic_functions.h"
__attribute__((unused)) static inline unsigned atomicDec(unsigned *address, unsigned val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
# 137
{ } 
#endif
# 139 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/device_atomic_functions.h"
__attribute__((unused)) static inline int atomicAnd(int *address, int val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
# 139
{ } 
#endif
# 141 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/device_atomic_functions.h"
__attribute__((unused)) static inline unsigned atomicAnd(unsigned *address, unsigned val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
# 141
{ } 
#endif
# 143 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/device_atomic_functions.h"
__attribute__((unused)) static inline int atomicOr(int *address, int val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
# 143
{ } 
#endif
# 145 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/device_atomic_functions.h"
__attribute__((unused)) static inline unsigned atomicOr(unsigned *address, unsigned val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
# 145
{ } 
#endif
# 147 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/device_atomic_functions.h"
__attribute__((unused)) static inline int atomicXor(int *address, int val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
# 147
{ } 
#endif
# 149 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/device_atomic_functions.h"
__attribute__((unused)) static inline unsigned atomicXor(unsigned *address, unsigned val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
# 149
{ } 
#endif
# 151 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/device_atomic_functions.h"
__attribute__((unused)) static inline int atomicCAS(int *address, int compare, int val) {int volatile ___ = 1;(void)address;(void)compare;(void)val;::exit(___);}
#if 0
# 151
{ } 
#endif
# 153 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/device_atomic_functions.h"
__attribute__((unused)) static inline unsigned atomicCAS(unsigned *address, unsigned compare, unsigned val) {int volatile ___ = 1;(void)address;(void)compare;(void)val;::exit(___);}
#if 0
# 153
{ } 
#endif
# 178 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/device_atomic_functions.h"
extern "C" {
# 187
}
# 196 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/device_atomic_functions.h"
__attribute__((unused)) static inline unsigned long long atomicAdd(unsigned long long *address, unsigned long long val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
# 196
{ } 
#endif
# 198 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/device_atomic_functions.h"
__attribute__((unused)) static inline unsigned long long atomicExch(unsigned long long *address, unsigned long long val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
# 198
{ } 
#endif
# 200 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/device_atomic_functions.h"
__attribute__((unused)) static inline unsigned long long atomicCAS(unsigned long long *address, unsigned long long compare, unsigned long long val) {int volatile ___ = 1;(void)address;(void)compare;(void)val;::exit(___);}
#if 0
# 200
{ } 
#endif
# 202 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/device_atomic_functions.h"
__attribute((deprecated("__any() is deprecated in favor of __any_sync() and may be removed in a future release (Use -Wno-deprecated-declarations to suppr" "ess this warning)."))) __attribute__((unused)) static inline bool any(bool cond) {int volatile ___ = 1;(void)cond;::exit(___);}
#if 0
# 202
{ } 
#endif
# 204 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/device_atomic_functions.h"
__attribute((deprecated("__all() is deprecated in favor of __all_sync() and may be removed in a future release (Use -Wno-deprecated-declarations to suppr" "ess this warning)."))) __attribute__((unused)) static inline bool all(bool cond) {int volatile ___ = 1;(void)cond;::exit(___);}
#if 0
# 204
{ } 
#endif
# 87 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/crt/device_double_functions.h"
extern "C" {
# 1139 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/crt/device_double_functions.h"
}
# 1147
__attribute__((unused)) static inline double fma(double a, double b, double c, cudaRoundMode mode); 
# 1149
__attribute__((unused)) static inline double dmul(double a, double b, cudaRoundMode mode = cudaRoundNearest); 
# 1151
__attribute__((unused)) static inline double dadd(double a, double b, cudaRoundMode mode = cudaRoundNearest); 
# 1153
__attribute__((unused)) static inline double dsub(double a, double b, cudaRoundMode mode = cudaRoundNearest); 
# 1155
__attribute__((unused)) static inline int double2int(double a, cudaRoundMode mode = cudaRoundZero); 
# 1157
__attribute__((unused)) static inline unsigned double2uint(double a, cudaRoundMode mode = cudaRoundZero); 
# 1159
__attribute__((unused)) static inline long long double2ll(double a, cudaRoundMode mode = cudaRoundZero); 
# 1161
__attribute__((unused)) static inline unsigned long long double2ull(double a, cudaRoundMode mode = cudaRoundZero); 
# 1163
__attribute__((unused)) static inline double ll2double(long long a, cudaRoundMode mode = cudaRoundNearest); 
# 1165
__attribute__((unused)) static inline double ull2double(unsigned long long a, cudaRoundMode mode = cudaRoundNearest); 
# 1167
__attribute__((unused)) static inline double int2double(int a, cudaRoundMode mode = cudaRoundNearest); 
# 1169
__attribute__((unused)) static inline double uint2double(unsigned a, cudaRoundMode mode = cudaRoundNearest); 
# 1171
__attribute__((unused)) static inline double float2double(float a, cudaRoundMode mode = cudaRoundNearest); 
# 93 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/crt/device_double_functions.hpp"
__attribute__((unused)) static inline double fma(double a, double b, double c, cudaRoundMode mode) 
# 94
{int volatile ___ = 1;(void)a;(void)b;(void)c;(void)mode;
# 99
::exit(___);}
#if 0
# 94
{ 
# 95
return (mode == (cudaRoundZero)) ? __fma_rz(a, b, c) : ((mode == (cudaRoundPosInf)) ? __fma_ru(a, b, c) : ((mode == (cudaRoundMinInf)) ? __fma_rd(a, b, c) : __fma_rn(a, b, c))); 
# 99
} 
#endif
# 101 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/crt/device_double_functions.hpp"
__attribute__((unused)) static inline double dmul(double a, double b, cudaRoundMode mode) 
# 102
{int volatile ___ = 1;(void)a;(void)b;(void)mode;
# 107
::exit(___);}
#if 0
# 102
{ 
# 103
return (mode == (cudaRoundZero)) ? __dmul_rz(a, b) : ((mode == (cudaRoundPosInf)) ? __dmul_ru(a, b) : ((mode == (cudaRoundMinInf)) ? __dmul_rd(a, b) : __dmul_rn(a, b))); 
# 107
} 
#endif
# 109 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/crt/device_double_functions.hpp"
__attribute__((unused)) static inline double dadd(double a, double b, cudaRoundMode mode) 
# 110
{int volatile ___ = 1;(void)a;(void)b;(void)mode;
# 115
::exit(___);}
#if 0
# 110
{ 
# 111
return (mode == (cudaRoundZero)) ? __dadd_rz(a, b) : ((mode == (cudaRoundPosInf)) ? __dadd_ru(a, b) : ((mode == (cudaRoundMinInf)) ? __dadd_rd(a, b) : __dadd_rn(a, b))); 
# 115
} 
#endif
# 117 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/crt/device_double_functions.hpp"
__attribute__((unused)) static inline double dsub(double a, double b, cudaRoundMode mode) 
# 118
{int volatile ___ = 1;(void)a;(void)b;(void)mode;
# 123
::exit(___);}
#if 0
# 118
{ 
# 119
return (mode == (cudaRoundZero)) ? __dsub_rz(a, b) : ((mode == (cudaRoundPosInf)) ? __dsub_ru(a, b) : ((mode == (cudaRoundMinInf)) ? __dsub_rd(a, b) : __dsub_rn(a, b))); 
# 123
} 
#endif
# 125 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/crt/device_double_functions.hpp"
__attribute__((unused)) static inline int double2int(double a, cudaRoundMode mode) 
# 126
{int volatile ___ = 1;(void)a;(void)mode;
# 131
::exit(___);}
#if 0
# 126
{ 
# 127
return (mode == (cudaRoundNearest)) ? __double2int_rn(a) : ((mode == (cudaRoundPosInf)) ? __double2int_ru(a) : ((mode == (cudaRoundMinInf)) ? __double2int_rd(a) : __double2int_rz(a))); 
# 131
} 
#endif
# 133 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/crt/device_double_functions.hpp"
__attribute__((unused)) static inline unsigned double2uint(double a, cudaRoundMode mode) 
# 134
{int volatile ___ = 1;(void)a;(void)mode;
# 139
::exit(___);}
#if 0
# 134
{ 
# 135
return (mode == (cudaRoundNearest)) ? __double2uint_rn(a) : ((mode == (cudaRoundPosInf)) ? __double2uint_ru(a) : ((mode == (cudaRoundMinInf)) ? __double2uint_rd(a) : __double2uint_rz(a))); 
# 139
} 
#endif
# 141 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/crt/device_double_functions.hpp"
__attribute__((unused)) static inline long long double2ll(double a, cudaRoundMode mode) 
# 142
{int volatile ___ = 1;(void)a;(void)mode;
# 147
::exit(___);}
#if 0
# 142
{ 
# 143
return (mode == (cudaRoundNearest)) ? __double2ll_rn(a) : ((mode == (cudaRoundPosInf)) ? __double2ll_ru(a) : ((mode == (cudaRoundMinInf)) ? __double2ll_rd(a) : __double2ll_rz(a))); 
# 147
} 
#endif
# 149 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/crt/device_double_functions.hpp"
__attribute__((unused)) static inline unsigned long long double2ull(double a, cudaRoundMode mode) 
# 150
{int volatile ___ = 1;(void)a;(void)mode;
# 155
::exit(___);}
#if 0
# 150
{ 
# 151
return (mode == (cudaRoundNearest)) ? __double2ull_rn(a) : ((mode == (cudaRoundPosInf)) ? __double2ull_ru(a) : ((mode == (cudaRoundMinInf)) ? __double2ull_rd(a) : __double2ull_rz(a))); 
# 155
} 
#endif
# 157 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/crt/device_double_functions.hpp"
__attribute__((unused)) static inline double ll2double(long long a, cudaRoundMode mode) 
# 158
{int volatile ___ = 1;(void)a;(void)mode;
# 163
::exit(___);}
#if 0
# 158
{ 
# 159
return (mode == (cudaRoundZero)) ? __ll2double_rz(a) : ((mode == (cudaRoundPosInf)) ? __ll2double_ru(a) : ((mode == (cudaRoundMinInf)) ? __ll2double_rd(a) : __ll2double_rn(a))); 
# 163
} 
#endif
# 165 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/crt/device_double_functions.hpp"
__attribute__((unused)) static inline double ull2double(unsigned long long a, cudaRoundMode mode) 
# 166
{int volatile ___ = 1;(void)a;(void)mode;
# 171
::exit(___);}
#if 0
# 166
{ 
# 167
return (mode == (cudaRoundZero)) ? __ull2double_rz(a) : ((mode == (cudaRoundPosInf)) ? __ull2double_ru(a) : ((mode == (cudaRoundMinInf)) ? __ull2double_rd(a) : __ull2double_rn(a))); 
# 171
} 
#endif
# 173 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/crt/device_double_functions.hpp"
__attribute__((unused)) static inline double int2double(int a, cudaRoundMode mode) 
# 174
{int volatile ___ = 1;(void)a;(void)mode;
# 176
::exit(___);}
#if 0
# 174
{ 
# 175
return (double)a; 
# 176
} 
#endif
# 178 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/crt/device_double_functions.hpp"
__attribute__((unused)) static inline double uint2double(unsigned a, cudaRoundMode mode) 
# 179
{int volatile ___ = 1;(void)a;(void)mode;
# 181
::exit(___);}
#if 0
# 179
{ 
# 180
return (double)a; 
# 181
} 
#endif
# 183 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/crt/device_double_functions.hpp"
__attribute__((unused)) static inline double float2double(float a, cudaRoundMode mode) 
# 184
{int volatile ___ = 1;(void)a;(void)mode;
# 186
::exit(___);}
#if 0
# 184
{ 
# 185
return (double)a; 
# 186
} 
#endif
# 106 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/sm_20_atomic_functions.h"
__attribute__((unused)) static inline float atomicAdd(float *address, float val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
# 106
{ } 
#endif
# 110 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/sm_32_atomic_functions.h"
__attribute__((unused)) static inline long long atomicMin(long long *address, long long val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
# 110
{ } 
#endif
# 112 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/sm_32_atomic_functions.h"
__attribute__((unused)) static inline long long atomicMax(long long *address, long long val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
# 112
{ } 
#endif
# 114 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/sm_32_atomic_functions.h"
__attribute__((unused)) static inline long long atomicAnd(long long *address, long long val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
# 114
{ } 
#endif
# 116 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/sm_32_atomic_functions.h"
__attribute__((unused)) static inline long long atomicOr(long long *address, long long val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
# 116
{ } 
#endif
# 118 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/sm_32_atomic_functions.h"
__attribute__((unused)) static inline long long atomicXor(long long *address, long long val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
# 118
{ } 
#endif
# 120 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/sm_32_atomic_functions.h"
__attribute__((unused)) static inline unsigned long long atomicMin(unsigned long long *address, unsigned long long val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
# 120
{ } 
#endif
# 122 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/sm_32_atomic_functions.h"
__attribute__((unused)) static inline unsigned long long atomicMax(unsigned long long *address, unsigned long long val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
# 122
{ } 
#endif
# 124 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/sm_32_atomic_functions.h"
__attribute__((unused)) static inline unsigned long long atomicAnd(unsigned long long *address, unsigned long long val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
# 124
{ } 
#endif
# 126 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/sm_32_atomic_functions.h"
__attribute__((unused)) static inline unsigned long long atomicOr(unsigned long long *address, unsigned long long val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
# 126
{ } 
#endif
# 128 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/sm_32_atomic_functions.h"
__attribute__((unused)) static inline unsigned long long atomicXor(unsigned long long *address, unsigned long long val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
# 128
{ } 
#endif
# 310 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/sm_60_atomic_functions.h"
__attribute__((unused)) static inline double atomicAdd(double *address, double val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
# 310
{ } 
#endif
# 313 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/sm_60_atomic_functions.h"
__attribute__((unused)) static inline int atomicAdd_block(int *address, int val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
# 313
{ } 
#endif
# 316 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/sm_60_atomic_functions.h"
__attribute__((unused)) static inline int atomicAdd_system(int *address, int val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
# 316
{ } 
#endif
# 319 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/sm_60_atomic_functions.h"
__attribute__((unused)) static inline unsigned atomicAdd_block(unsigned *address, unsigned val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
# 319
{ } 
#endif
# 322 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/sm_60_atomic_functions.h"
__attribute__((unused)) static inline unsigned atomicAdd_system(unsigned *address, unsigned val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
# 322
{ } 
#endif
# 325 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/sm_60_atomic_functions.h"
__attribute__((unused)) static inline unsigned long long atomicAdd_block(unsigned long long *address, unsigned long long val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
# 325
{ } 
#endif
# 328 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/sm_60_atomic_functions.h"
__attribute__((unused)) static inline unsigned long long atomicAdd_system(unsigned long long *address, unsigned long long val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
# 328
{ } 
#endif
# 331 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/sm_60_atomic_functions.h"
__attribute__((unused)) static inline float atomicAdd_block(float *address, float val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
# 331
{ } 
#endif
# 334 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/sm_60_atomic_functions.h"
__attribute__((unused)) static inline float atomicAdd_system(float *address, float val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
# 334
{ } 
#endif
# 337 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/sm_60_atomic_functions.h"
__attribute__((unused)) static inline double atomicAdd_block(double *address, double val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
# 337
{ } 
#endif
# 340 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/sm_60_atomic_functions.h"
__attribute__((unused)) static inline double atomicAdd_system(double *address, double val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
# 340
{ } 
#endif
# 343 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/sm_60_atomic_functions.h"
__attribute__((unused)) static inline int atomicSub_block(int *address, int val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
# 343
{ } 
#endif
# 346 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/sm_60_atomic_functions.h"
__attribute__((unused)) static inline int atomicSub_system(int *address, int val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
# 346
{ } 
#endif
# 349 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/sm_60_atomic_functions.h"
__attribute__((unused)) static inline unsigned atomicSub_block(unsigned *address, unsigned val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
# 349
{ } 
#endif
# 352 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/sm_60_atomic_functions.h"
__attribute__((unused)) static inline unsigned atomicSub_system(unsigned *address, unsigned val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
# 352
{ } 
#endif
# 355 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/sm_60_atomic_functions.h"
__attribute__((unused)) static inline int atomicExch_block(int *address, int val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
# 355
{ } 
#endif
# 358 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/sm_60_atomic_functions.h"
__attribute__((unused)) static inline int atomicExch_system(int *address, int val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
# 358
{ } 
#endif
# 361 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/sm_60_atomic_functions.h"
__attribute__((unused)) static inline unsigned atomicExch_block(unsigned *address, unsigned val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
# 361
{ } 
#endif
# 364 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/sm_60_atomic_functions.h"
__attribute__((unused)) static inline unsigned atomicExch_system(unsigned *address, unsigned val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
# 364
{ } 
#endif
# 367 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/sm_60_atomic_functions.h"
__attribute__((unused)) static inline unsigned long long atomicExch_block(unsigned long long *address, unsigned long long val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
# 367
{ } 
#endif
# 370 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/sm_60_atomic_functions.h"
__attribute__((unused)) static inline unsigned long long atomicExch_system(unsigned long long *address, unsigned long long val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
# 370
{ } 
#endif
# 373 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/sm_60_atomic_functions.h"
__attribute__((unused)) static inline float atomicExch_block(float *address, float val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
# 373
{ } 
#endif
# 376 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/sm_60_atomic_functions.h"
__attribute__((unused)) static inline float atomicExch_system(float *address, float val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
# 376
{ } 
#endif
# 379 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/sm_60_atomic_functions.h"
__attribute__((unused)) static inline int atomicMin_block(int *address, int val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
# 379
{ } 
#endif
# 382 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/sm_60_atomic_functions.h"
__attribute__((unused)) static inline int atomicMin_system(int *address, int val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
# 382
{ } 
#endif
# 385 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/sm_60_atomic_functions.h"
__attribute__((unused)) static inline long long atomicMin_block(long long *address, long long val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
# 385
{ } 
#endif
# 388 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/sm_60_atomic_functions.h"
__attribute__((unused)) static inline long long atomicMin_system(long long *address, long long val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
# 388
{ } 
#endif
# 391 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/sm_60_atomic_functions.h"
__attribute__((unused)) static inline unsigned atomicMin_block(unsigned *address, unsigned val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
# 391
{ } 
#endif
# 394 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/sm_60_atomic_functions.h"
__attribute__((unused)) static inline unsigned atomicMin_system(unsigned *address, unsigned val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
# 394
{ } 
#endif
# 397 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/sm_60_atomic_functions.h"
__attribute__((unused)) static inline unsigned long long atomicMin_block(unsigned long long *address, unsigned long long val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
# 397
{ } 
#endif
# 400 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/sm_60_atomic_functions.h"
__attribute__((unused)) static inline unsigned long long atomicMin_system(unsigned long long *address, unsigned long long val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
# 400
{ } 
#endif
# 403 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/sm_60_atomic_functions.h"
__attribute__((unused)) static inline int atomicMax_block(int *address, int val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
# 403
{ } 
#endif
# 406 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/sm_60_atomic_functions.h"
__attribute__((unused)) static inline int atomicMax_system(int *address, int val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
# 406
{ } 
#endif
# 409 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/sm_60_atomic_functions.h"
__attribute__((unused)) static inline long long atomicMax_block(long long *address, long long val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
# 409
{ } 
#endif
# 412 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/sm_60_atomic_functions.h"
__attribute__((unused)) static inline long long atomicMax_system(long long *address, long long val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
# 412
{ } 
#endif
# 415 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/sm_60_atomic_functions.h"
__attribute__((unused)) static inline unsigned atomicMax_block(unsigned *address, unsigned val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
# 415
{ } 
#endif
# 418 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/sm_60_atomic_functions.h"
__attribute__((unused)) static inline unsigned atomicMax_system(unsigned *address, unsigned val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
# 418
{ } 
#endif
# 421 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/sm_60_atomic_functions.h"
__attribute__((unused)) static inline unsigned long long atomicMax_block(unsigned long long *address, unsigned long long val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
# 421
{ } 
#endif
# 424 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/sm_60_atomic_functions.h"
__attribute__((unused)) static inline unsigned long long atomicMax_system(unsigned long long *address, unsigned long long val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
# 424
{ } 
#endif
# 427 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/sm_60_atomic_functions.h"
__attribute__((unused)) static inline unsigned atomicInc_block(unsigned *address, unsigned val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
# 427
{ } 
#endif
# 430 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/sm_60_atomic_functions.h"
__attribute__((unused)) static inline unsigned atomicInc_system(unsigned *address, unsigned val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
# 430
{ } 
#endif
# 433 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/sm_60_atomic_functions.h"
__attribute__((unused)) static inline unsigned atomicDec_block(unsigned *address, unsigned val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
# 433
{ } 
#endif
# 436 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/sm_60_atomic_functions.h"
__attribute__((unused)) static inline unsigned atomicDec_system(unsigned *address, unsigned val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
# 436
{ } 
#endif
# 439 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/sm_60_atomic_functions.h"
__attribute__((unused)) static inline int atomicCAS_block(int *address, int compare, int val) {int volatile ___ = 1;(void)address;(void)compare;(void)val;::exit(___);}
#if 0
# 439
{ } 
#endif
# 442 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/sm_60_atomic_functions.h"
__attribute__((unused)) static inline int atomicCAS_system(int *address, int compare, int val) {int volatile ___ = 1;(void)address;(void)compare;(void)val;::exit(___);}
#if 0
# 442
{ } 
#endif
# 445 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/sm_60_atomic_functions.h"
__attribute__((unused)) static inline unsigned atomicCAS_block(unsigned *address, unsigned compare, unsigned 
# 446
val) {int volatile ___ = 1;(void)address;(void)compare;(void)val;::exit(___);}
#if 0
# 446
{ } 
#endif
# 449 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/sm_60_atomic_functions.h"
__attribute__((unused)) static inline unsigned atomicCAS_system(unsigned *address, unsigned compare, unsigned 
# 450
val) {int volatile ___ = 1;(void)address;(void)compare;(void)val;::exit(___);}
#if 0
# 450
{ } 
#endif
# 453 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/sm_60_atomic_functions.h"
__attribute__((unused)) static inline unsigned long long atomicCAS_block(unsigned long long *address, unsigned long long 
# 454
compare, unsigned long long 
# 455
val) {int volatile ___ = 1;(void)address;(void)compare;(void)val;::exit(___);}
#if 0
# 455
{ } 
#endif
# 458 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/sm_60_atomic_functions.h"
__attribute__((unused)) static inline unsigned long long atomicCAS_system(unsigned long long *address, unsigned long long 
# 459
compare, unsigned long long 
# 460
val) {int volatile ___ = 1;(void)address;(void)compare;(void)val;::exit(___);}
#if 0
# 460
{ } 
#endif
# 463 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/sm_60_atomic_functions.h"
__attribute__((unused)) static inline int atomicAnd_block(int *address, int val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
# 463
{ } 
#endif
# 466 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/sm_60_atomic_functions.h"
__attribute__((unused)) static inline int atomicAnd_system(int *address, int val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
# 466
{ } 
#endif
# 469 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/sm_60_atomic_functions.h"
__attribute__((unused)) static inline long long atomicAnd_block(long long *address, long long val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
# 469
{ } 
#endif
# 472 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/sm_60_atomic_functions.h"
__attribute__((unused)) static inline long long atomicAnd_system(long long *address, long long val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
# 472
{ } 
#endif
# 475 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/sm_60_atomic_functions.h"
__attribute__((unused)) static inline unsigned atomicAnd_block(unsigned *address, unsigned val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
# 475
{ } 
#endif
# 478 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/sm_60_atomic_functions.h"
__attribute__((unused)) static inline unsigned atomicAnd_system(unsigned *address, unsigned val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
# 478
{ } 
#endif
# 481 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/sm_60_atomic_functions.h"
__attribute__((unused)) static inline unsigned long long atomicAnd_block(unsigned long long *address, unsigned long long val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
# 481
{ } 
#endif
# 484 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/sm_60_atomic_functions.h"
__attribute__((unused)) static inline unsigned long long atomicAnd_system(unsigned long long *address, unsigned long long val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
# 484
{ } 
#endif
# 487 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/sm_60_atomic_functions.h"
__attribute__((unused)) static inline int atomicOr_block(int *address, int val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
# 487
{ } 
#endif
# 490 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/sm_60_atomic_functions.h"
__attribute__((unused)) static inline int atomicOr_system(int *address, int val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
# 490
{ } 
#endif
# 493 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/sm_60_atomic_functions.h"
__attribute__((unused)) static inline long long atomicOr_block(long long *address, long long val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
# 493
{ } 
#endif
# 496 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/sm_60_atomic_functions.h"
__attribute__((unused)) static inline long long atomicOr_system(long long *address, long long val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
# 496
{ } 
#endif
# 499 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/sm_60_atomic_functions.h"
__attribute__((unused)) static inline unsigned atomicOr_block(unsigned *address, unsigned val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
# 499
{ } 
#endif
# 502 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/sm_60_atomic_functions.h"
__attribute__((unused)) static inline unsigned atomicOr_system(unsigned *address, unsigned val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
# 502
{ } 
#endif
# 505 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/sm_60_atomic_functions.h"
__attribute__((unused)) static inline unsigned long long atomicOr_block(unsigned long long *address, unsigned long long val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
# 505
{ } 
#endif
# 508 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/sm_60_atomic_functions.h"
__attribute__((unused)) static inline unsigned long long atomicOr_system(unsigned long long *address, unsigned long long val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
# 508
{ } 
#endif
# 511 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/sm_60_atomic_functions.h"
__attribute__((unused)) static inline int atomicXor_block(int *address, int val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
# 511
{ } 
#endif
# 514 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/sm_60_atomic_functions.h"
__attribute__((unused)) static inline int atomicXor_system(int *address, int val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
# 514
{ } 
#endif
# 517 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/sm_60_atomic_functions.h"
__attribute__((unused)) static inline long long atomicXor_block(long long *address, long long val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
# 517
{ } 
#endif
# 520 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/sm_60_atomic_functions.h"
__attribute__((unused)) static inline long long atomicXor_system(long long *address, long long val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
# 520
{ } 
#endif
# 523 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/sm_60_atomic_functions.h"
__attribute__((unused)) static inline unsigned atomicXor_block(unsigned *address, unsigned val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
# 523
{ } 
#endif
# 526 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/sm_60_atomic_functions.h"
__attribute__((unused)) static inline unsigned atomicXor_system(unsigned *address, unsigned val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
# 526
{ } 
#endif
# 529 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/sm_60_atomic_functions.h"
__attribute__((unused)) static inline unsigned long long atomicXor_block(unsigned long long *address, unsigned long long val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
# 529
{ } 
#endif
# 532 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/sm_60_atomic_functions.h"
__attribute__((unused)) static inline unsigned long long atomicXor_system(unsigned long long *address, unsigned long long val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
# 532
{ } 
#endif
# 95 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/sm_20_intrinsics.h"
extern "C" {
# 1508 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/sm_20_intrinsics.h"
}
# 1520 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/sm_20_intrinsics.h"
__attribute((deprecated("__ballot() is deprecated in favor of __ballot_sync() and may be removed in a future release (Use -Wno-deprecated-declarations to" " suppress this warning)."))) __attribute__((unused)) static inline unsigned ballot(bool pred) {int volatile ___ = 1;(void)pred;::exit(___);}
#if 0
# 1520
{ } 
#endif
# 1522 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/sm_20_intrinsics.h"
__attribute__((unused)) static inline int syncthreads_count(bool pred) {int volatile ___ = 1;(void)pred;::exit(___);}
#if 0
# 1522
{ } 
#endif
# 1524 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/sm_20_intrinsics.h"
__attribute__((unused)) static inline bool syncthreads_and(bool pred) {int volatile ___ = 1;(void)pred;::exit(___);}
#if 0
# 1524
{ } 
#endif
# 1526 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/sm_20_intrinsics.h"
__attribute__((unused)) static inline bool syncthreads_or(bool pred) {int volatile ___ = 1;(void)pred;::exit(___);}
#if 0
# 1526
{ } 
#endif
# 1531 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/sm_20_intrinsics.h"
__attribute__((unused)) static inline unsigned __isGlobal(const void *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 1531
{ } 
#endif
# 1532 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/sm_20_intrinsics.h"
__attribute__((unused)) static inline unsigned __isShared(const void *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 1532
{ } 
#endif
# 1533 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/sm_20_intrinsics.h"
__attribute__((unused)) static inline unsigned __isConstant(const void *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 1533
{ } 
#endif
# 1534 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/sm_20_intrinsics.h"
__attribute__((unused)) static inline unsigned __isLocal(const void *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 1534
{ } 
#endif
# 1536 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/sm_20_intrinsics.h"
__attribute__((unused)) static inline size_t __cvta_generic_to_global(const void *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 1536
{ } 
#endif
# 1537 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/sm_20_intrinsics.h"
__attribute__((unused)) static inline size_t __cvta_generic_to_shared(const void *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 1537
{ } 
#endif
# 1538 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/sm_20_intrinsics.h"
__attribute__((unused)) static inline size_t __cvta_generic_to_constant(const void *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 1538
{ } 
#endif
# 1539 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/sm_20_intrinsics.h"
__attribute__((unused)) static inline size_t __cvta_generic_to_local(const void *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 1539
{ } 
#endif
# 1541 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/sm_20_intrinsics.h"
__attribute__((unused)) static inline void *__cvta_global_to_generic(size_t rawbits) {int volatile ___ = 1;(void)rawbits;::exit(___);}
#if 0
# 1541
{ } 
#endif
# 1542 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/sm_20_intrinsics.h"
__attribute__((unused)) static inline void *__cvta_shared_to_generic(size_t rawbits) {int volatile ___ = 1;(void)rawbits;::exit(___);}
#if 0
# 1542
{ } 
#endif
# 1543 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/sm_20_intrinsics.h"
__attribute__((unused)) static inline void *__cvta_constant_to_generic(size_t rawbits) {int volatile ___ = 1;(void)rawbits;::exit(___);}
#if 0
# 1543
{ } 
#endif
# 1544 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/sm_20_intrinsics.h"
__attribute__((unused)) static inline void *__cvta_local_to_generic(size_t rawbits) {int volatile ___ = 1;(void)rawbits;::exit(___);}
#if 0
# 1544
{ } 
#endif
# 109 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/sm_30_intrinsics.h"
__attribute__((unused)) static inline unsigned __fns(unsigned mask, unsigned base, int offset) {int volatile ___ = 1;(void)mask;(void)base;(void)offset;::exit(___);}
#if 0
# 109
{ } 
#endif
# 110 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/sm_30_intrinsics.h"
__attribute__((unused)) static inline void __barrier_sync(unsigned id) {int volatile ___ = 1;(void)id;::exit(___);}
#if 0
# 110
{ } 
#endif
# 111 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/sm_30_intrinsics.h"
__attribute__((unused)) static inline void __barrier_sync_count(unsigned id, unsigned cnt) {int volatile ___ = 1;(void)id;(void)cnt;::exit(___);}
#if 0
# 111
{ } 
#endif
# 112 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/sm_30_intrinsics.h"
__attribute__((unused)) static inline void __syncwarp(unsigned mask = 4294967295U) {int volatile ___ = 1;(void)mask;::exit(___);}
#if 0
# 112
{ } 
#endif
# 113 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/sm_30_intrinsics.h"
__attribute__((unused)) static inline int __all_sync(unsigned mask, int pred) {int volatile ___ = 1;(void)mask;(void)pred;::exit(___);}
#if 0
# 113
{ } 
#endif
# 114 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/sm_30_intrinsics.h"
__attribute__((unused)) static inline int __any_sync(unsigned mask, int pred) {int volatile ___ = 1;(void)mask;(void)pred;::exit(___);}
#if 0
# 114
{ } 
#endif
# 115 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/sm_30_intrinsics.h"
__attribute__((unused)) static inline int __uni_sync(unsigned mask, int pred) {int volatile ___ = 1;(void)mask;(void)pred;::exit(___);}
#if 0
# 115
{ } 
#endif
# 116 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/sm_30_intrinsics.h"
__attribute__((unused)) static inline unsigned __ballot_sync(unsigned mask, int pred) {int volatile ___ = 1;(void)mask;(void)pred;::exit(___);}
#if 0
# 116
{ } 
#endif
# 117 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/sm_30_intrinsics.h"
__attribute__((unused)) static inline unsigned __activemask() {int volatile ___ = 1;::exit(___);}
#if 0
# 117
{ } 
#endif
# 126 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/sm_30_intrinsics.h"
__attribute((deprecated("__shfl() is deprecated in favor of __shfl_sync() and may be removed in a future release (Use -Wno-deprecated-declarations to sup" "press this warning)."))) __attribute__((unused)) static inline int __shfl(int var, int srcLane, int width = 32) {int volatile ___ = 1;(void)var;(void)srcLane;(void)width;::exit(___);}
#if 0
# 126
{ } 
#endif
# 127 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/sm_30_intrinsics.h"
__attribute((deprecated("__shfl() is deprecated in favor of __shfl_sync() and may be removed in a future release (Use -Wno-deprecated-declarations to sup" "press this warning)."))) __attribute__((unused)) static inline unsigned __shfl(unsigned var, int srcLane, int width = 32) {int volatile ___ = 1;(void)var;(void)srcLane;(void)width;::exit(___);}
#if 0
# 127
{ } 
#endif
# 128 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/sm_30_intrinsics.h"
__attribute((deprecated("__shfl_up() is deprecated in favor of __shfl_up_sync() and may be removed in a future release (Use -Wno-deprecated-declarations " "to suppress this warning)."))) __attribute__((unused)) static inline int __shfl_up(int var, unsigned delta, int width = 32) {int volatile ___ = 1;(void)var;(void)delta;(void)width;::exit(___);}
#if 0
# 128
{ } 
#endif
# 129 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/sm_30_intrinsics.h"
__attribute((deprecated("__shfl_up() is deprecated in favor of __shfl_up_sync() and may be removed in a future release (Use -Wno-deprecated-declarations " "to suppress this warning)."))) __attribute__((unused)) static inline unsigned __shfl_up(unsigned var, unsigned delta, int width = 32) {int volatile ___ = 1;(void)var;(void)delta;(void)width;::exit(___);}
#if 0
# 129
{ } 
#endif
# 130 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/sm_30_intrinsics.h"
__attribute((deprecated("__shfl_down() is deprecated in favor of __shfl_down_sync() and may be removed in a future release (Use -Wno-deprecated-declarati" "ons to suppress this warning)."))) __attribute__((unused)) static inline int __shfl_down(int var, unsigned delta, int width = 32) {int volatile ___ = 1;(void)var;(void)delta;(void)width;::exit(___);}
#if 0
# 130
{ } 
#endif
# 131 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/sm_30_intrinsics.h"
__attribute((deprecated("__shfl_down() is deprecated in favor of __shfl_down_sync() and may be removed in a future release (Use -Wno-deprecated-declarati" "ons to suppress this warning)."))) __attribute__((unused)) static inline unsigned __shfl_down(unsigned var, unsigned delta, int width = 32) {int volatile ___ = 1;(void)var;(void)delta;(void)width;::exit(___);}
#if 0
# 131
{ } 
#endif
# 132 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/sm_30_intrinsics.h"
__attribute((deprecated("__shfl_xor() is deprecated in favor of __shfl_xor_sync() and may be removed in a future release (Use -Wno-deprecated-declaration" "s to suppress this warning)."))) __attribute__((unused)) static inline int __shfl_xor(int var, int laneMask, int width = 32) {int volatile ___ = 1;(void)var;(void)laneMask;(void)width;::exit(___);}
#if 0
# 132
{ } 
#endif
# 133 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/sm_30_intrinsics.h"
__attribute((deprecated("__shfl_xor() is deprecated in favor of __shfl_xor_sync() and may be removed in a future release (Use -Wno-deprecated-declaration" "s to suppress this warning)."))) __attribute__((unused)) static inline unsigned __shfl_xor(unsigned var, int laneMask, int width = 32) {int volatile ___ = 1;(void)var;(void)laneMask;(void)width;::exit(___);}
#if 0
# 133
{ } 
#endif
# 134 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/sm_30_intrinsics.h"
__attribute((deprecated("__shfl() is deprecated in favor of __shfl_sync() and may be removed in a future release (Use -Wno-deprecated-declarations to sup" "press this warning)."))) __attribute__((unused)) static inline float __shfl(float var, int srcLane, int width = 32) {int volatile ___ = 1;(void)var;(void)srcLane;(void)width;::exit(___);}
#if 0
# 134
{ } 
#endif
# 135 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/sm_30_intrinsics.h"
__attribute((deprecated("__shfl_up() is deprecated in favor of __shfl_up_sync() and may be removed in a future release (Use -Wno-deprecated-declarations " "to suppress this warning)."))) __attribute__((unused)) static inline float __shfl_up(float var, unsigned delta, int width = 32) {int volatile ___ = 1;(void)var;(void)delta;(void)width;::exit(___);}
#if 0
# 135
{ } 
#endif
# 136 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/sm_30_intrinsics.h"
__attribute((deprecated("__shfl_down() is deprecated in favor of __shfl_down_sync() and may be removed in a future release (Use -Wno-deprecated-declarati" "ons to suppress this warning)."))) __attribute__((unused)) static inline float __shfl_down(float var, unsigned delta, int width = 32) {int volatile ___ = 1;(void)var;(void)delta;(void)width;::exit(___);}
#if 0
# 136
{ } 
#endif
# 137 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/sm_30_intrinsics.h"
__attribute((deprecated("__shfl_xor() is deprecated in favor of __shfl_xor_sync() and may be removed in a future release (Use -Wno-deprecated-declaration" "s to suppress this warning)."))) __attribute__((unused)) static inline float __shfl_xor(float var, int laneMask, int width = 32) {int volatile ___ = 1;(void)var;(void)laneMask;(void)width;::exit(___);}
#if 0
# 137
{ } 
#endif
# 140 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/sm_30_intrinsics.h"
__attribute__((unused)) static inline int __shfl_sync(unsigned mask, int var, int srcLane, int width = 32) {int volatile ___ = 1;(void)mask;(void)var;(void)srcLane;(void)width;::exit(___);}
#if 0
# 140
{ } 
#endif
# 141 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/sm_30_intrinsics.h"
__attribute__((unused)) static inline unsigned __shfl_sync(unsigned mask, unsigned var, int srcLane, int width = 32) {int volatile ___ = 1;(void)mask;(void)var;(void)srcLane;(void)width;::exit(___);}
#if 0
# 141
{ } 
#endif
# 142 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/sm_30_intrinsics.h"
__attribute__((unused)) static inline int __shfl_up_sync(unsigned mask, int var, unsigned delta, int width = 32) {int volatile ___ = 1;(void)mask;(void)var;(void)delta;(void)width;::exit(___);}
#if 0
# 142
{ } 
#endif
# 143 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/sm_30_intrinsics.h"
__attribute__((unused)) static inline unsigned __shfl_up_sync(unsigned mask, unsigned var, unsigned delta, int width = 32) {int volatile ___ = 1;(void)mask;(void)var;(void)delta;(void)width;::exit(___);}
#if 0
# 143
{ } 
#endif
# 144 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/sm_30_intrinsics.h"
__attribute__((unused)) static inline int __shfl_down_sync(unsigned mask, int var, unsigned delta, int width = 32) {int volatile ___ = 1;(void)mask;(void)var;(void)delta;(void)width;::exit(___);}
#if 0
# 144
{ } 
#endif
# 145 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/sm_30_intrinsics.h"
__attribute__((unused)) static inline unsigned __shfl_down_sync(unsigned mask, unsigned var, unsigned delta, int width = 32) {int volatile ___ = 1;(void)mask;(void)var;(void)delta;(void)width;::exit(___);}
#if 0
# 145
{ } 
#endif
# 146 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/sm_30_intrinsics.h"
__attribute__((unused)) static inline int __shfl_xor_sync(unsigned mask, int var, int laneMask, int width = 32) {int volatile ___ = 1;(void)mask;(void)var;(void)laneMask;(void)width;::exit(___);}
#if 0
# 146
{ } 
#endif
# 147 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/sm_30_intrinsics.h"
__attribute__((unused)) static inline unsigned __shfl_xor_sync(unsigned mask, unsigned var, int laneMask, int width = 32) {int volatile ___ = 1;(void)mask;(void)var;(void)laneMask;(void)width;::exit(___);}
#if 0
# 147
{ } 
#endif
# 148 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/sm_30_intrinsics.h"
__attribute__((unused)) static inline float __shfl_sync(unsigned mask, float var, int srcLane, int width = 32) {int volatile ___ = 1;(void)mask;(void)var;(void)srcLane;(void)width;::exit(___);}
#if 0
# 148
{ } 
#endif
# 149 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/sm_30_intrinsics.h"
__attribute__((unused)) static inline float __shfl_up_sync(unsigned mask, float var, unsigned delta, int width = 32) {int volatile ___ = 1;(void)mask;(void)var;(void)delta;(void)width;::exit(___);}
#if 0
# 149
{ } 
#endif
# 150 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/sm_30_intrinsics.h"
__attribute__((unused)) static inline float __shfl_down_sync(unsigned mask, float var, unsigned delta, int width = 32) {int volatile ___ = 1;(void)mask;(void)var;(void)delta;(void)width;::exit(___);}
#if 0
# 150
{ } 
#endif
# 151 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/sm_30_intrinsics.h"
__attribute__((unused)) static inline float __shfl_xor_sync(unsigned mask, float var, int laneMask, int width = 32) {int volatile ___ = 1;(void)mask;(void)var;(void)laneMask;(void)width;::exit(___);}
#if 0
# 151
{ } 
#endif
# 155 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/sm_30_intrinsics.h"
__attribute((deprecated("__shfl() is deprecated in favor of __shfl_sync() and may be removed in a future release (Use -Wno-deprecated-declarations to sup" "press this warning)."))) __attribute__((unused)) static inline unsigned long long __shfl(unsigned long long var, int srcLane, int width = 32) {int volatile ___ = 1;(void)var;(void)srcLane;(void)width;::exit(___);}
#if 0
# 155
{ } 
#endif
# 156 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/sm_30_intrinsics.h"
__attribute((deprecated("__shfl() is deprecated in favor of __shfl_sync() and may be removed in a future release (Use -Wno-deprecated-declarations to sup" "press this warning)."))) __attribute__((unused)) static inline long long __shfl(long long var, int srcLane, int width = 32) {int volatile ___ = 1;(void)var;(void)srcLane;(void)width;::exit(___);}
#if 0
# 156
{ } 
#endif
# 157 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/sm_30_intrinsics.h"
__attribute((deprecated("__shfl_up() is deprecated in favor of __shfl_up_sync() and may be removed in a future release (Use -Wno-deprecated-declarations " "to suppress this warning)."))) __attribute__((unused)) static inline long long __shfl_up(long long var, unsigned delta, int width = 32) {int volatile ___ = 1;(void)var;(void)delta;(void)width;::exit(___);}
#if 0
# 157
{ } 
#endif
# 158 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/sm_30_intrinsics.h"
__attribute((deprecated("__shfl_up() is deprecated in favor of __shfl_up_sync() and may be removed in a future release (Use -Wno-deprecated-declarations " "to suppress this warning)."))) __attribute__((unused)) static inline unsigned long long __shfl_up(unsigned long long var, unsigned delta, int width = 32) {int volatile ___ = 1;(void)var;(void)delta;(void)width;::exit(___);}
#if 0
# 158
{ } 
#endif
# 159 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/sm_30_intrinsics.h"
__attribute((deprecated("__shfl_down() is deprecated in favor of __shfl_down_sync() and may be removed in a future release (Use -Wno-deprecated-declarati" "ons to suppress this warning)."))) __attribute__((unused)) static inline long long __shfl_down(long long var, unsigned delta, int width = 32) {int volatile ___ = 1;(void)var;(void)delta;(void)width;::exit(___);}
#if 0
# 159
{ } 
#endif
# 160 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/sm_30_intrinsics.h"
__attribute((deprecated("__shfl_down() is deprecated in favor of __shfl_down_sync() and may be removed in a future release (Use -Wno-deprecated-declarati" "ons to suppress this warning)."))) __attribute__((unused)) static inline unsigned long long __shfl_down(unsigned long long var, unsigned delta, int width = 32) {int volatile ___ = 1;(void)var;(void)delta;(void)width;::exit(___);}
#if 0
# 160
{ } 
#endif
# 161 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/sm_30_intrinsics.h"
__attribute((deprecated("__shfl_xor() is deprecated in favor of __shfl_xor_sync() and may be removed in a future release (Use -Wno-deprecated-declaration" "s to suppress this warning)."))) __attribute__((unused)) static inline long long __shfl_xor(long long var, int laneMask, int width = 32) {int volatile ___ = 1;(void)var;(void)laneMask;(void)width;::exit(___);}
#if 0
# 161
{ } 
#endif
# 162 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/sm_30_intrinsics.h"
__attribute((deprecated("__shfl_xor() is deprecated in favor of __shfl_xor_sync() and may be removed in a future release (Use -Wno-deprecated-declaration" "s to suppress this warning)."))) __attribute__((unused)) static inline unsigned long long __shfl_xor(unsigned long long var, int laneMask, int width = 32) {int volatile ___ = 1;(void)var;(void)laneMask;(void)width;::exit(___);}
#if 0
# 162
{ } 
#endif
# 163 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/sm_30_intrinsics.h"
__attribute((deprecated("__shfl() is deprecated in favor of __shfl_sync() and may be removed in a future release (Use -Wno-deprecated-declarations to sup" "press this warning)."))) __attribute__((unused)) static inline double __shfl(double var, int srcLane, int width = 32) {int volatile ___ = 1;(void)var;(void)srcLane;(void)width;::exit(___);}
#if 0
# 163
{ } 
#endif
# 164 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/sm_30_intrinsics.h"
__attribute((deprecated("__shfl_up() is deprecated in favor of __shfl_up_sync() and may be removed in a future release (Use -Wno-deprecated-declarations " "to suppress this warning)."))) __attribute__((unused)) static inline double __shfl_up(double var, unsigned delta, int width = 32) {int volatile ___ = 1;(void)var;(void)delta;(void)width;::exit(___);}
#if 0
# 164
{ } 
#endif
# 165 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/sm_30_intrinsics.h"
__attribute((deprecated("__shfl_down() is deprecated in favor of __shfl_down_sync() and may be removed in a future release (Use -Wno-deprecated-declarati" "ons to suppress this warning)."))) __attribute__((unused)) static inline double __shfl_down(double var, unsigned delta, int width = 32) {int volatile ___ = 1;(void)var;(void)delta;(void)width;::exit(___);}
#if 0
# 165
{ } 
#endif
# 166 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/sm_30_intrinsics.h"
__attribute((deprecated("__shfl_xor() is deprecated in favor of __shfl_xor_sync() and may be removed in a future release (Use -Wno-deprecated-declaration" "s to suppress this warning)."))) __attribute__((unused)) static inline double __shfl_xor(double var, int laneMask, int width = 32) {int volatile ___ = 1;(void)var;(void)laneMask;(void)width;::exit(___);}
#if 0
# 166
{ } 
#endif
# 169 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/sm_30_intrinsics.h"
__attribute__((unused)) static inline long long __shfl_sync(unsigned mask, long long var, int srcLane, int width = 32) {int volatile ___ = 1;(void)mask;(void)var;(void)srcLane;(void)width;::exit(___);}
#if 0
# 169
{ } 
#endif
# 170 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/sm_30_intrinsics.h"
__attribute__((unused)) static inline unsigned long long __shfl_sync(unsigned mask, unsigned long long var, int srcLane, int width = 32) {int volatile ___ = 1;(void)mask;(void)var;(void)srcLane;(void)width;::exit(___);}
#if 0
# 170
{ } 
#endif
# 171 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/sm_30_intrinsics.h"
__attribute__((unused)) static inline long long __shfl_up_sync(unsigned mask, long long var, unsigned delta, int width = 32) {int volatile ___ = 1;(void)mask;(void)var;(void)delta;(void)width;::exit(___);}
#if 0
# 171
{ } 
#endif
# 172 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/sm_30_intrinsics.h"
__attribute__((unused)) static inline unsigned long long __shfl_up_sync(unsigned mask, unsigned long long var, unsigned delta, int width = 32) {int volatile ___ = 1;(void)mask;(void)var;(void)delta;(void)width;::exit(___);}
#if 0
# 172
{ } 
#endif
# 173 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/sm_30_intrinsics.h"
__attribute__((unused)) static inline long long __shfl_down_sync(unsigned mask, long long var, unsigned delta, int width = 32) {int volatile ___ = 1;(void)mask;(void)var;(void)delta;(void)width;::exit(___);}
#if 0
# 173
{ } 
#endif
# 174 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/sm_30_intrinsics.h"
__attribute__((unused)) static inline unsigned long long __shfl_down_sync(unsigned mask, unsigned long long var, unsigned delta, int width = 32) {int volatile ___ = 1;(void)mask;(void)var;(void)delta;(void)width;::exit(___);}
#if 0
# 174
{ } 
#endif
# 175 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/sm_30_intrinsics.h"
__attribute__((unused)) static inline long long __shfl_xor_sync(unsigned mask, long long var, int laneMask, int width = 32) {int volatile ___ = 1;(void)mask;(void)var;(void)laneMask;(void)width;::exit(___);}
#if 0
# 175
{ } 
#endif
# 176 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/sm_30_intrinsics.h"
__attribute__((unused)) static inline unsigned long long __shfl_xor_sync(unsigned mask, unsigned long long var, int laneMask, int width = 32) {int volatile ___ = 1;(void)mask;(void)var;(void)laneMask;(void)width;::exit(___);}
#if 0
# 176
{ } 
#endif
# 177 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/sm_30_intrinsics.h"
__attribute__((unused)) static inline double __shfl_sync(unsigned mask, double var, int srcLane, int width = 32) {int volatile ___ = 1;(void)mask;(void)var;(void)srcLane;(void)width;::exit(___);}
#if 0
# 177
{ } 
#endif
# 178 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/sm_30_intrinsics.h"
__attribute__((unused)) static inline double __shfl_up_sync(unsigned mask, double var, unsigned delta, int width = 32) {int volatile ___ = 1;(void)mask;(void)var;(void)delta;(void)width;::exit(___);}
#if 0
# 178
{ } 
#endif
# 179 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/sm_30_intrinsics.h"
__attribute__((unused)) static inline double __shfl_down_sync(unsigned mask, double var, unsigned delta, int width = 32) {int volatile ___ = 1;(void)mask;(void)var;(void)delta;(void)width;::exit(___);}
#if 0
# 179
{ } 
#endif
# 180 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/sm_30_intrinsics.h"
__attribute__((unused)) static inline double __shfl_xor_sync(unsigned mask, double var, int laneMask, int width = 32) {int volatile ___ = 1;(void)mask;(void)var;(void)laneMask;(void)width;::exit(___);}
#if 0
# 180
{ } 
#endif
# 184 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/sm_30_intrinsics.h"
__attribute((deprecated("__shfl() is deprecated in favor of __shfl_sync() and may be removed in a future release (Use -Wno-deprecated-declarations to sup" "press this warning)."))) __attribute__((unused)) static inline long __shfl(long var, int srcLane, int width = 32) {int volatile ___ = 1;(void)var;(void)srcLane;(void)width;::exit(___);}
#if 0
# 184
{ } 
#endif
# 185 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/sm_30_intrinsics.h"
__attribute((deprecated("__shfl() is deprecated in favor of __shfl_sync() and may be removed in a future release (Use -Wno-deprecated-declarations to sup" "press this warning)."))) __attribute__((unused)) static inline unsigned long __shfl(unsigned long var, int srcLane, int width = 32) {int volatile ___ = 1;(void)var;(void)srcLane;(void)width;::exit(___);}
#if 0
# 185
{ } 
#endif
# 186 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/sm_30_intrinsics.h"
__attribute((deprecated("__shfl_up() is deprecated in favor of __shfl_up_sync() and may be removed in a future release (Use -Wno-deprecated-declarations " "to suppress this warning)."))) __attribute__((unused)) static inline long __shfl_up(long var, unsigned delta, int width = 32) {int volatile ___ = 1;(void)var;(void)delta;(void)width;::exit(___);}
#if 0
# 186
{ } 
#endif
# 187 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/sm_30_intrinsics.h"
__attribute((deprecated("__shfl_up() is deprecated in favor of __shfl_up_sync() and may be removed in a future release (Use -Wno-deprecated-declarations " "to suppress this warning)."))) __attribute__((unused)) static inline unsigned long __shfl_up(unsigned long var, unsigned delta, int width = 32) {int volatile ___ = 1;(void)var;(void)delta;(void)width;::exit(___);}
#if 0
# 187
{ } 
#endif
# 188 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/sm_30_intrinsics.h"
__attribute((deprecated("__shfl_down() is deprecated in favor of __shfl_down_sync() and may be removed in a future release (Use -Wno-deprecated-declarati" "ons to suppress this warning)."))) __attribute__((unused)) static inline long __shfl_down(long var, unsigned delta, int width = 32) {int volatile ___ = 1;(void)var;(void)delta;(void)width;::exit(___);}
#if 0
# 188
{ } 
#endif
# 189 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/sm_30_intrinsics.h"
__attribute((deprecated("__shfl_down() is deprecated in favor of __shfl_down_sync() and may be removed in a future release (Use -Wno-deprecated-declarati" "ons to suppress this warning)."))) __attribute__((unused)) static inline unsigned long __shfl_down(unsigned long var, unsigned delta, int width = 32) {int volatile ___ = 1;(void)var;(void)delta;(void)width;::exit(___);}
#if 0
# 189
{ } 
#endif
# 190 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/sm_30_intrinsics.h"
__attribute((deprecated("__shfl_xor() is deprecated in favor of __shfl_xor_sync() and may be removed in a future release (Use -Wno-deprecated-declaration" "s to suppress this warning)."))) __attribute__((unused)) static inline long __shfl_xor(long var, int laneMask, int width = 32) {int volatile ___ = 1;(void)var;(void)laneMask;(void)width;::exit(___);}
#if 0
# 190
{ } 
#endif
# 191 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/sm_30_intrinsics.h"
__attribute((deprecated("__shfl_xor() is deprecated in favor of __shfl_xor_sync() and may be removed in a future release (Use -Wno-deprecated-declaration" "s to suppress this warning)."))) __attribute__((unused)) static inline unsigned long __shfl_xor(unsigned long var, int laneMask, int width = 32) {int volatile ___ = 1;(void)var;(void)laneMask;(void)width;::exit(___);}
#if 0
# 191
{ } 
#endif
# 194 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/sm_30_intrinsics.h"
__attribute__((unused)) static inline long __shfl_sync(unsigned mask, long var, int srcLane, int width = 32) {int volatile ___ = 1;(void)mask;(void)var;(void)srcLane;(void)width;::exit(___);}
#if 0
# 194
{ } 
#endif
# 195 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/sm_30_intrinsics.h"
__attribute__((unused)) static inline unsigned long __shfl_sync(unsigned mask, unsigned long var, int srcLane, int width = 32) {int volatile ___ = 1;(void)mask;(void)var;(void)srcLane;(void)width;::exit(___);}
#if 0
# 195
{ } 
#endif
# 196 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/sm_30_intrinsics.h"
__attribute__((unused)) static inline long __shfl_up_sync(unsigned mask, long var, unsigned delta, int width = 32) {int volatile ___ = 1;(void)mask;(void)var;(void)delta;(void)width;::exit(___);}
#if 0
# 196
{ } 
#endif
# 197 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/sm_30_intrinsics.h"
__attribute__((unused)) static inline unsigned long __shfl_up_sync(unsigned mask, unsigned long var, unsigned delta, int width = 32) {int volatile ___ = 1;(void)mask;(void)var;(void)delta;(void)width;::exit(___);}
#if 0
# 197
{ } 
#endif
# 198 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/sm_30_intrinsics.h"
__attribute__((unused)) static inline long __shfl_down_sync(unsigned mask, long var, unsigned delta, int width = 32) {int volatile ___ = 1;(void)mask;(void)var;(void)delta;(void)width;::exit(___);}
#if 0
# 198
{ } 
#endif
# 199 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/sm_30_intrinsics.h"
__attribute__((unused)) static inline unsigned long __shfl_down_sync(unsigned mask, unsigned long var, unsigned delta, int width = 32) {int volatile ___ = 1;(void)mask;(void)var;(void)delta;(void)width;::exit(___);}
#if 0
# 199
{ } 
#endif
# 200 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/sm_30_intrinsics.h"
__attribute__((unused)) static inline long __shfl_xor_sync(unsigned mask, long var, int laneMask, int width = 32) {int volatile ___ = 1;(void)mask;(void)var;(void)laneMask;(void)width;::exit(___);}
#if 0
# 200
{ } 
#endif
# 201 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/sm_30_intrinsics.h"
__attribute__((unused)) static inline unsigned long __shfl_xor_sync(unsigned mask, unsigned long var, int laneMask, int width = 32) {int volatile ___ = 1;(void)mask;(void)var;(void)laneMask;(void)width;::exit(___);}
#if 0
# 201
{ } 
#endif
# 87 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline long __ldg(const long *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 87
{ } 
#endif
# 88 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline unsigned long __ldg(const unsigned long *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 88
{ } 
#endif
# 90 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline char __ldg(const char *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 90
{ } 
#endif
# 91 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline signed char __ldg(const signed char *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 91
{ } 
#endif
# 92 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline short __ldg(const short *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 92
{ } 
#endif
# 93 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline int __ldg(const int *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 93
{ } 
#endif
# 94 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline long long __ldg(const long long *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 94
{ } 
#endif
# 95 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline char2 __ldg(const char2 *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 95
{ } 
#endif
# 96 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline char4 __ldg(const char4 *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 96
{ } 
#endif
# 97 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline short2 __ldg(const short2 *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 97
{ } 
#endif
# 98 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline short4 __ldg(const short4 *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 98
{ } 
#endif
# 99 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline int2 __ldg(const int2 *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 99
{ } 
#endif
# 100 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline int4 __ldg(const int4 *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 100
{ } 
#endif
# 101 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline longlong2 __ldg(const longlong2 *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 101
{ } 
#endif
# 103 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline unsigned char __ldg(const unsigned char *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 103
{ } 
#endif
# 104 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline unsigned short __ldg(const unsigned short *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 104
{ } 
#endif
# 105 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline unsigned __ldg(const unsigned *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 105
{ } 
#endif
# 106 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline unsigned long long __ldg(const unsigned long long *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 106
{ } 
#endif
# 107 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline uchar2 __ldg(const uchar2 *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 107
{ } 
#endif
# 108 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline uchar4 __ldg(const uchar4 *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 108
{ } 
#endif
# 109 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline ushort2 __ldg(const ushort2 *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 109
{ } 
#endif
# 110 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline ushort4 __ldg(const ushort4 *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 110
{ } 
#endif
# 111 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline uint2 __ldg(const uint2 *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 111
{ } 
#endif
# 112 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline uint4 __ldg(const uint4 *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 112
{ } 
#endif
# 113 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline ulonglong2 __ldg(const ulonglong2 *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 113
{ } 
#endif
# 115 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline float __ldg(const float *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 115
{ } 
#endif
# 116 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline double __ldg(const double *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 116
{ } 
#endif
# 117 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline float2 __ldg(const float2 *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 117
{ } 
#endif
# 118 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline float4 __ldg(const float4 *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 118
{ } 
#endif
# 119 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline double2 __ldg(const double2 *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 119
{ } 
#endif
# 123 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline long __ldcg(const long *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 123
{ } 
#endif
# 124 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline unsigned long __ldcg(const unsigned long *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 124
{ } 
#endif
# 126 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline char __ldcg(const char *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 126
{ } 
#endif
# 127 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline signed char __ldcg(const signed char *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 127
{ } 
#endif
# 128 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline short __ldcg(const short *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 128
{ } 
#endif
# 129 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline int __ldcg(const int *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 129
{ } 
#endif
# 130 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline long long __ldcg(const long long *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 130
{ } 
#endif
# 131 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline char2 __ldcg(const char2 *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 131
{ } 
#endif
# 132 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline char4 __ldcg(const char4 *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 132
{ } 
#endif
# 133 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline short2 __ldcg(const short2 *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 133
{ } 
#endif
# 134 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline short4 __ldcg(const short4 *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 134
{ } 
#endif
# 135 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline int2 __ldcg(const int2 *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 135
{ } 
#endif
# 136 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline int4 __ldcg(const int4 *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 136
{ } 
#endif
# 137 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline longlong2 __ldcg(const longlong2 *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 137
{ } 
#endif
# 139 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline unsigned char __ldcg(const unsigned char *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 139
{ } 
#endif
# 140 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline unsigned short __ldcg(const unsigned short *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 140
{ } 
#endif
# 141 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline unsigned __ldcg(const unsigned *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 141
{ } 
#endif
# 142 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline unsigned long long __ldcg(const unsigned long long *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 142
{ } 
#endif
# 143 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline uchar2 __ldcg(const uchar2 *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 143
{ } 
#endif
# 144 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline uchar4 __ldcg(const uchar4 *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 144
{ } 
#endif
# 145 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline ushort2 __ldcg(const ushort2 *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 145
{ } 
#endif
# 146 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline ushort4 __ldcg(const ushort4 *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 146
{ } 
#endif
# 147 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline uint2 __ldcg(const uint2 *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 147
{ } 
#endif
# 148 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline uint4 __ldcg(const uint4 *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 148
{ } 
#endif
# 149 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline ulonglong2 __ldcg(const ulonglong2 *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 149
{ } 
#endif
# 151 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline float __ldcg(const float *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 151
{ } 
#endif
# 152 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline double __ldcg(const double *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 152
{ } 
#endif
# 153 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline float2 __ldcg(const float2 *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 153
{ } 
#endif
# 154 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline float4 __ldcg(const float4 *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 154
{ } 
#endif
# 155 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline double2 __ldcg(const double2 *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 155
{ } 
#endif
# 159 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline long __ldca(const long *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 159
{ } 
#endif
# 160 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline unsigned long __ldca(const unsigned long *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 160
{ } 
#endif
# 162 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline char __ldca(const char *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 162
{ } 
#endif
# 163 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline signed char __ldca(const signed char *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 163
{ } 
#endif
# 164 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline short __ldca(const short *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 164
{ } 
#endif
# 165 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline int __ldca(const int *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 165
{ } 
#endif
# 166 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline long long __ldca(const long long *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 166
{ } 
#endif
# 167 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline char2 __ldca(const char2 *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 167
{ } 
#endif
# 168 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline char4 __ldca(const char4 *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 168
{ } 
#endif
# 169 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline short2 __ldca(const short2 *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 169
{ } 
#endif
# 170 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline short4 __ldca(const short4 *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 170
{ } 
#endif
# 171 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline int2 __ldca(const int2 *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 171
{ } 
#endif
# 172 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline int4 __ldca(const int4 *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 172
{ } 
#endif
# 173 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline longlong2 __ldca(const longlong2 *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 173
{ } 
#endif
# 175 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline unsigned char __ldca(const unsigned char *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 175
{ } 
#endif
# 176 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline unsigned short __ldca(const unsigned short *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 176
{ } 
#endif
# 177 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline unsigned __ldca(const unsigned *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 177
{ } 
#endif
# 178 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline unsigned long long __ldca(const unsigned long long *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 178
{ } 
#endif
# 179 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline uchar2 __ldca(const uchar2 *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 179
{ } 
#endif
# 180 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline uchar4 __ldca(const uchar4 *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 180
{ } 
#endif
# 181 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline ushort2 __ldca(const ushort2 *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 181
{ } 
#endif
# 182 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline ushort4 __ldca(const ushort4 *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 182
{ } 
#endif
# 183 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline uint2 __ldca(const uint2 *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 183
{ } 
#endif
# 184 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline uint4 __ldca(const uint4 *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 184
{ } 
#endif
# 185 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline ulonglong2 __ldca(const ulonglong2 *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 185
{ } 
#endif
# 187 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline float __ldca(const float *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 187
{ } 
#endif
# 188 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline double __ldca(const double *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 188
{ } 
#endif
# 189 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline float2 __ldca(const float2 *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 189
{ } 
#endif
# 190 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline float4 __ldca(const float4 *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 190
{ } 
#endif
# 191 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline double2 __ldca(const double2 *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 191
{ } 
#endif
# 195 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline long __ldcs(const long *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 195
{ } 
#endif
# 196 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline unsigned long __ldcs(const unsigned long *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 196
{ } 
#endif
# 198 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline char __ldcs(const char *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 198
{ } 
#endif
# 199 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline signed char __ldcs(const signed char *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 199
{ } 
#endif
# 200 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline short __ldcs(const short *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 200
{ } 
#endif
# 201 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline int __ldcs(const int *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 201
{ } 
#endif
# 202 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline long long __ldcs(const long long *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 202
{ } 
#endif
# 203 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline char2 __ldcs(const char2 *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 203
{ } 
#endif
# 204 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline char4 __ldcs(const char4 *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 204
{ } 
#endif
# 205 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline short2 __ldcs(const short2 *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 205
{ } 
#endif
# 206 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline short4 __ldcs(const short4 *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 206
{ } 
#endif
# 207 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline int2 __ldcs(const int2 *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 207
{ } 
#endif
# 208 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline int4 __ldcs(const int4 *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 208
{ } 
#endif
# 209 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline longlong2 __ldcs(const longlong2 *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 209
{ } 
#endif
# 211 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline unsigned char __ldcs(const unsigned char *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 211
{ } 
#endif
# 212 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline unsigned short __ldcs(const unsigned short *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 212
{ } 
#endif
# 213 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline unsigned __ldcs(const unsigned *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 213
{ } 
#endif
# 214 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline unsigned long long __ldcs(const unsigned long long *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 214
{ } 
#endif
# 215 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline uchar2 __ldcs(const uchar2 *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 215
{ } 
#endif
# 216 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline uchar4 __ldcs(const uchar4 *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 216
{ } 
#endif
# 217 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline ushort2 __ldcs(const ushort2 *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 217
{ } 
#endif
# 218 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline ushort4 __ldcs(const ushort4 *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 218
{ } 
#endif
# 219 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline uint2 __ldcs(const uint2 *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 219
{ } 
#endif
# 220 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline uint4 __ldcs(const uint4 *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 220
{ } 
#endif
# 221 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline ulonglong2 __ldcs(const ulonglong2 *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 221
{ } 
#endif
# 223 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline float __ldcs(const float *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 223
{ } 
#endif
# 224 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline double __ldcs(const double *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 224
{ } 
#endif
# 225 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline float2 __ldcs(const float2 *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 225
{ } 
#endif
# 226 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline float4 __ldcs(const float4 *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 226
{ } 
#endif
# 227 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline double2 __ldcs(const double2 *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 227
{ } 
#endif
# 231 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline long __ldlu(const long *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 231
{ } 
#endif
# 232 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline unsigned long __ldlu(const unsigned long *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 232
{ } 
#endif
# 234 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline char __ldlu(const char *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 234
{ } 
#endif
# 235 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline signed char __ldlu(const signed char *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 235
{ } 
#endif
# 236 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline short __ldlu(const short *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 236
{ } 
#endif
# 237 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline int __ldlu(const int *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 237
{ } 
#endif
# 238 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline long long __ldlu(const long long *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 238
{ } 
#endif
# 239 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline char2 __ldlu(const char2 *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 239
{ } 
#endif
# 240 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline char4 __ldlu(const char4 *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 240
{ } 
#endif
# 241 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline short2 __ldlu(const short2 *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 241
{ } 
#endif
# 242 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline short4 __ldlu(const short4 *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 242
{ } 
#endif
# 243 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline int2 __ldlu(const int2 *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 243
{ } 
#endif
# 244 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline int4 __ldlu(const int4 *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 244
{ } 
#endif
# 245 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline longlong2 __ldlu(const longlong2 *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 245
{ } 
#endif
# 247 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline unsigned char __ldlu(const unsigned char *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 247
{ } 
#endif
# 248 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline unsigned short __ldlu(const unsigned short *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 248
{ } 
#endif
# 249 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline unsigned __ldlu(const unsigned *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 249
{ } 
#endif
# 250 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline unsigned long long __ldlu(const unsigned long long *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 250
{ } 
#endif
# 251 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline uchar2 __ldlu(const uchar2 *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 251
{ } 
#endif
# 252 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline uchar4 __ldlu(const uchar4 *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 252
{ } 
#endif
# 253 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline ushort2 __ldlu(const ushort2 *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 253
{ } 
#endif
# 254 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline ushort4 __ldlu(const ushort4 *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 254
{ } 
#endif
# 255 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline uint2 __ldlu(const uint2 *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 255
{ } 
#endif
# 256 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline uint4 __ldlu(const uint4 *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 256
{ } 
#endif
# 257 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline ulonglong2 __ldlu(const ulonglong2 *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 257
{ } 
#endif
# 259 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline float __ldlu(const float *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 259
{ } 
#endif
# 260 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline double __ldlu(const double *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 260
{ } 
#endif
# 261 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline float2 __ldlu(const float2 *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 261
{ } 
#endif
# 262 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline float4 __ldlu(const float4 *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 262
{ } 
#endif
# 263 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline double2 __ldlu(const double2 *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 263
{ } 
#endif
# 267 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline long __ldcv(const long *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 267
{ } 
#endif
# 268 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline unsigned long __ldcv(const unsigned long *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 268
{ } 
#endif
# 270 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline char __ldcv(const char *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 270
{ } 
#endif
# 271 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline signed char __ldcv(const signed char *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 271
{ } 
#endif
# 272 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline short __ldcv(const short *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 272
{ } 
#endif
# 273 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline int __ldcv(const int *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 273
{ } 
#endif
# 274 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline long long __ldcv(const long long *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 274
{ } 
#endif
# 275 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline char2 __ldcv(const char2 *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 275
{ } 
#endif
# 276 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline char4 __ldcv(const char4 *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 276
{ } 
#endif
# 277 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline short2 __ldcv(const short2 *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 277
{ } 
#endif
# 278 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline short4 __ldcv(const short4 *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 278
{ } 
#endif
# 279 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline int2 __ldcv(const int2 *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 279
{ } 
#endif
# 280 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline int4 __ldcv(const int4 *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 280
{ } 
#endif
# 281 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline longlong2 __ldcv(const longlong2 *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 281
{ } 
#endif
# 283 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline unsigned char __ldcv(const unsigned char *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 283
{ } 
#endif
# 284 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline unsigned short __ldcv(const unsigned short *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 284
{ } 
#endif
# 285 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline unsigned __ldcv(const unsigned *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 285
{ } 
#endif
# 286 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline unsigned long long __ldcv(const unsigned long long *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 286
{ } 
#endif
# 287 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline uchar2 __ldcv(const uchar2 *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 287
{ } 
#endif
# 288 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline uchar4 __ldcv(const uchar4 *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 288
{ } 
#endif
# 289 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline ushort2 __ldcv(const ushort2 *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 289
{ } 
#endif
# 290 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline ushort4 __ldcv(const ushort4 *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 290
{ } 
#endif
# 291 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline uint2 __ldcv(const uint2 *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 291
{ } 
#endif
# 292 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline uint4 __ldcv(const uint4 *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 292
{ } 
#endif
# 293 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline ulonglong2 __ldcv(const ulonglong2 *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 293
{ } 
#endif
# 295 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline float __ldcv(const float *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 295
{ } 
#endif
# 296 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline double __ldcv(const double *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 296
{ } 
#endif
# 297 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline float2 __ldcv(const float2 *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 297
{ } 
#endif
# 298 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline float4 __ldcv(const float4 *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 298
{ } 
#endif
# 299 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline double2 __ldcv(const double2 *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 299
{ } 
#endif
# 303 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline void __stwb(long *ptr, long value) {int volatile ___ = 1;(void)ptr;(void)value;::exit(___);}
#if 0
# 303
{ } 
#endif
# 304 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline void __stwb(unsigned long *ptr, unsigned long value) {int volatile ___ = 1;(void)ptr;(void)value;::exit(___);}
#if 0
# 304
{ } 
#endif
# 306 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline void __stwb(char *ptr, char value) {int volatile ___ = 1;(void)ptr;(void)value;::exit(___);}
#if 0
# 306
{ } 
#endif
# 307 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline void __stwb(signed char *ptr, signed char value) {int volatile ___ = 1;(void)ptr;(void)value;::exit(___);}
#if 0
# 307
{ } 
#endif
# 308 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline void __stwb(short *ptr, short value) {int volatile ___ = 1;(void)ptr;(void)value;::exit(___);}
#if 0
# 308
{ } 
#endif
# 309 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline void __stwb(int *ptr, int value) {int volatile ___ = 1;(void)ptr;(void)value;::exit(___);}
#if 0
# 309
{ } 
#endif
# 310 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline void __stwb(long long *ptr, long long value) {int volatile ___ = 1;(void)ptr;(void)value;::exit(___);}
#if 0
# 310
{ } 
#endif
# 311 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline void __stwb(char2 *ptr, char2 value) {int volatile ___ = 1;(void)ptr;(void)value;::exit(___);}
#if 0
# 311
{ } 
#endif
# 312 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline void __stwb(char4 *ptr, char4 value) {int volatile ___ = 1;(void)ptr;(void)value;::exit(___);}
#if 0
# 312
{ } 
#endif
# 313 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline void __stwb(short2 *ptr, short2 value) {int volatile ___ = 1;(void)ptr;(void)value;::exit(___);}
#if 0
# 313
{ } 
#endif
# 314 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline void __stwb(short4 *ptr, short4 value) {int volatile ___ = 1;(void)ptr;(void)value;::exit(___);}
#if 0
# 314
{ } 
#endif
# 315 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline void __stwb(int2 *ptr, int2 value) {int volatile ___ = 1;(void)ptr;(void)value;::exit(___);}
#if 0
# 315
{ } 
#endif
# 316 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline void __stwb(int4 *ptr, int4 value) {int volatile ___ = 1;(void)ptr;(void)value;::exit(___);}
#if 0
# 316
{ } 
#endif
# 317 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline void __stwb(longlong2 *ptr, longlong2 value) {int volatile ___ = 1;(void)ptr;(void)value;::exit(___);}
#if 0
# 317
{ } 
#endif
# 319 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline void __stwb(unsigned char *ptr, unsigned char value) {int volatile ___ = 1;(void)ptr;(void)value;::exit(___);}
#if 0
# 319
{ } 
#endif
# 320 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline void __stwb(unsigned short *ptr, unsigned short value) {int volatile ___ = 1;(void)ptr;(void)value;::exit(___);}
#if 0
# 320
{ } 
#endif
# 321 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline void __stwb(unsigned *ptr, unsigned value) {int volatile ___ = 1;(void)ptr;(void)value;::exit(___);}
#if 0
# 321
{ } 
#endif
# 322 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline void __stwb(unsigned long long *ptr, unsigned long long value) {int volatile ___ = 1;(void)ptr;(void)value;::exit(___);}
#if 0
# 322
{ } 
#endif
# 323 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline void __stwb(uchar2 *ptr, uchar2 value) {int volatile ___ = 1;(void)ptr;(void)value;::exit(___);}
#if 0
# 323
{ } 
#endif
# 324 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline void __stwb(uchar4 *ptr, uchar4 value) {int volatile ___ = 1;(void)ptr;(void)value;::exit(___);}
#if 0
# 324
{ } 
#endif
# 325 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline void __stwb(ushort2 *ptr, ushort2 value) {int volatile ___ = 1;(void)ptr;(void)value;::exit(___);}
#if 0
# 325
{ } 
#endif
# 326 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline void __stwb(ushort4 *ptr, ushort4 value) {int volatile ___ = 1;(void)ptr;(void)value;::exit(___);}
#if 0
# 326
{ } 
#endif
# 327 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline void __stwb(uint2 *ptr, uint2 value) {int volatile ___ = 1;(void)ptr;(void)value;::exit(___);}
#if 0
# 327
{ } 
#endif
# 328 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline void __stwb(uint4 *ptr, uint4 value) {int volatile ___ = 1;(void)ptr;(void)value;::exit(___);}
#if 0
# 328
{ } 
#endif
# 329 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline void __stwb(ulonglong2 *ptr, ulonglong2 value) {int volatile ___ = 1;(void)ptr;(void)value;::exit(___);}
#if 0
# 329
{ } 
#endif
# 331 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline void __stwb(float *ptr, float value) {int volatile ___ = 1;(void)ptr;(void)value;::exit(___);}
#if 0
# 331
{ } 
#endif
# 332 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline void __stwb(double *ptr, double value) {int volatile ___ = 1;(void)ptr;(void)value;::exit(___);}
#if 0
# 332
{ } 
#endif
# 333 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline void __stwb(float2 *ptr, float2 value) {int volatile ___ = 1;(void)ptr;(void)value;::exit(___);}
#if 0
# 333
{ } 
#endif
# 334 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline void __stwb(float4 *ptr, float4 value) {int volatile ___ = 1;(void)ptr;(void)value;::exit(___);}
#if 0
# 334
{ } 
#endif
# 335 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline void __stwb(double2 *ptr, double2 value) {int volatile ___ = 1;(void)ptr;(void)value;::exit(___);}
#if 0
# 335
{ } 
#endif
# 339 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline void __stcg(long *ptr, long value) {int volatile ___ = 1;(void)ptr;(void)value;::exit(___);}
#if 0
# 339
{ } 
#endif
# 340 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline void __stcg(unsigned long *ptr, unsigned long value) {int volatile ___ = 1;(void)ptr;(void)value;::exit(___);}
#if 0
# 340
{ } 
#endif
# 342 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline void __stcg(char *ptr, char value) {int volatile ___ = 1;(void)ptr;(void)value;::exit(___);}
#if 0
# 342
{ } 
#endif
# 343 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline void __stcg(signed char *ptr, signed char value) {int volatile ___ = 1;(void)ptr;(void)value;::exit(___);}
#if 0
# 343
{ } 
#endif
# 344 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline void __stcg(short *ptr, short value) {int volatile ___ = 1;(void)ptr;(void)value;::exit(___);}
#if 0
# 344
{ } 
#endif
# 345 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline void __stcg(int *ptr, int value) {int volatile ___ = 1;(void)ptr;(void)value;::exit(___);}
#if 0
# 345
{ } 
#endif
# 346 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline void __stcg(long long *ptr, long long value) {int volatile ___ = 1;(void)ptr;(void)value;::exit(___);}
#if 0
# 346
{ } 
#endif
# 347 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline void __stcg(char2 *ptr, char2 value) {int volatile ___ = 1;(void)ptr;(void)value;::exit(___);}
#if 0
# 347
{ } 
#endif
# 348 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline void __stcg(char4 *ptr, char4 value) {int volatile ___ = 1;(void)ptr;(void)value;::exit(___);}
#if 0
# 348
{ } 
#endif
# 349 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline void __stcg(short2 *ptr, short2 value) {int volatile ___ = 1;(void)ptr;(void)value;::exit(___);}
#if 0
# 349
{ } 
#endif
# 350 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline void __stcg(short4 *ptr, short4 value) {int volatile ___ = 1;(void)ptr;(void)value;::exit(___);}
#if 0
# 350
{ } 
#endif
# 351 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline void __stcg(int2 *ptr, int2 value) {int volatile ___ = 1;(void)ptr;(void)value;::exit(___);}
#if 0
# 351
{ } 
#endif
# 352 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline void __stcg(int4 *ptr, int4 value) {int volatile ___ = 1;(void)ptr;(void)value;::exit(___);}
#if 0
# 352
{ } 
#endif
# 353 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline void __stcg(longlong2 *ptr, longlong2 value) {int volatile ___ = 1;(void)ptr;(void)value;::exit(___);}
#if 0
# 353
{ } 
#endif
# 355 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline void __stcg(unsigned char *ptr, unsigned char value) {int volatile ___ = 1;(void)ptr;(void)value;::exit(___);}
#if 0
# 355
{ } 
#endif
# 356 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline void __stcg(unsigned short *ptr, unsigned short value) {int volatile ___ = 1;(void)ptr;(void)value;::exit(___);}
#if 0
# 356
{ } 
#endif
# 357 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline void __stcg(unsigned *ptr, unsigned value) {int volatile ___ = 1;(void)ptr;(void)value;::exit(___);}
#if 0
# 357
{ } 
#endif
# 358 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline void __stcg(unsigned long long *ptr, unsigned long long value) {int volatile ___ = 1;(void)ptr;(void)value;::exit(___);}
#if 0
# 358
{ } 
#endif
# 359 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline void __stcg(uchar2 *ptr, uchar2 value) {int volatile ___ = 1;(void)ptr;(void)value;::exit(___);}
#if 0
# 359
{ } 
#endif
# 360 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline void __stcg(uchar4 *ptr, uchar4 value) {int volatile ___ = 1;(void)ptr;(void)value;::exit(___);}
#if 0
# 360
{ } 
#endif
# 361 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline void __stcg(ushort2 *ptr, ushort2 value) {int volatile ___ = 1;(void)ptr;(void)value;::exit(___);}
#if 0
# 361
{ } 
#endif
# 362 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline void __stcg(ushort4 *ptr, ushort4 value) {int volatile ___ = 1;(void)ptr;(void)value;::exit(___);}
#if 0
# 362
{ } 
#endif
# 363 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline void __stcg(uint2 *ptr, uint2 value) {int volatile ___ = 1;(void)ptr;(void)value;::exit(___);}
#if 0
# 363
{ } 
#endif
# 364 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline void __stcg(uint4 *ptr, uint4 value) {int volatile ___ = 1;(void)ptr;(void)value;::exit(___);}
#if 0
# 364
{ } 
#endif
# 365 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline void __stcg(ulonglong2 *ptr, ulonglong2 value) {int volatile ___ = 1;(void)ptr;(void)value;::exit(___);}
#if 0
# 365
{ } 
#endif
# 367 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline void __stcg(float *ptr, float value) {int volatile ___ = 1;(void)ptr;(void)value;::exit(___);}
#if 0
# 367
{ } 
#endif
# 368 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline void __stcg(double *ptr, double value) {int volatile ___ = 1;(void)ptr;(void)value;::exit(___);}
#if 0
# 368
{ } 
#endif
# 369 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline void __stcg(float2 *ptr, float2 value) {int volatile ___ = 1;(void)ptr;(void)value;::exit(___);}
#if 0
# 369
{ } 
#endif
# 370 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline void __stcg(float4 *ptr, float4 value) {int volatile ___ = 1;(void)ptr;(void)value;::exit(___);}
#if 0
# 370
{ } 
#endif
# 371 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline void __stcg(double2 *ptr, double2 value) {int volatile ___ = 1;(void)ptr;(void)value;::exit(___);}
#if 0
# 371
{ } 
#endif
# 375 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline void __stcs(long *ptr, long value) {int volatile ___ = 1;(void)ptr;(void)value;::exit(___);}
#if 0
# 375
{ } 
#endif
# 376 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline void __stcs(unsigned long *ptr, unsigned long value) {int volatile ___ = 1;(void)ptr;(void)value;::exit(___);}
#if 0
# 376
{ } 
#endif
# 378 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline void __stcs(char *ptr, char value) {int volatile ___ = 1;(void)ptr;(void)value;::exit(___);}
#if 0
# 378
{ } 
#endif
# 379 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline void __stcs(signed char *ptr, signed char value) {int volatile ___ = 1;(void)ptr;(void)value;::exit(___);}
#if 0
# 379
{ } 
#endif
# 380 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline void __stcs(short *ptr, short value) {int volatile ___ = 1;(void)ptr;(void)value;::exit(___);}
#if 0
# 380
{ } 
#endif
# 381 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline void __stcs(int *ptr, int value) {int volatile ___ = 1;(void)ptr;(void)value;::exit(___);}
#if 0
# 381
{ } 
#endif
# 382 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline void __stcs(long long *ptr, long long value) {int volatile ___ = 1;(void)ptr;(void)value;::exit(___);}
#if 0
# 382
{ } 
#endif
# 383 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline void __stcs(char2 *ptr, char2 value) {int volatile ___ = 1;(void)ptr;(void)value;::exit(___);}
#if 0
# 383
{ } 
#endif
# 384 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline void __stcs(char4 *ptr, char4 value) {int volatile ___ = 1;(void)ptr;(void)value;::exit(___);}
#if 0
# 384
{ } 
#endif
# 385 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline void __stcs(short2 *ptr, short2 value) {int volatile ___ = 1;(void)ptr;(void)value;::exit(___);}
#if 0
# 385
{ } 
#endif
# 386 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline void __stcs(short4 *ptr, short4 value) {int volatile ___ = 1;(void)ptr;(void)value;::exit(___);}
#if 0
# 386
{ } 
#endif
# 387 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline void __stcs(int2 *ptr, int2 value) {int volatile ___ = 1;(void)ptr;(void)value;::exit(___);}
#if 0
# 387
{ } 
#endif
# 388 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline void __stcs(int4 *ptr, int4 value) {int volatile ___ = 1;(void)ptr;(void)value;::exit(___);}
#if 0
# 388
{ } 
#endif
# 389 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline void __stcs(longlong2 *ptr, longlong2 value) {int volatile ___ = 1;(void)ptr;(void)value;::exit(___);}
#if 0
# 389
{ } 
#endif
# 391 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline void __stcs(unsigned char *ptr, unsigned char value) {int volatile ___ = 1;(void)ptr;(void)value;::exit(___);}
#if 0
# 391
{ } 
#endif
# 392 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline void __stcs(unsigned short *ptr, unsigned short value) {int volatile ___ = 1;(void)ptr;(void)value;::exit(___);}
#if 0
# 392
{ } 
#endif
# 393 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline void __stcs(unsigned *ptr, unsigned value) {int volatile ___ = 1;(void)ptr;(void)value;::exit(___);}
#if 0
# 393
{ } 
#endif
# 394 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline void __stcs(unsigned long long *ptr, unsigned long long value) {int volatile ___ = 1;(void)ptr;(void)value;::exit(___);}
#if 0
# 394
{ } 
#endif
# 395 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline void __stcs(uchar2 *ptr, uchar2 value) {int volatile ___ = 1;(void)ptr;(void)value;::exit(___);}
#if 0
# 395
{ } 
#endif
# 396 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline void __stcs(uchar4 *ptr, uchar4 value) {int volatile ___ = 1;(void)ptr;(void)value;::exit(___);}
#if 0
# 396
{ } 
#endif
# 397 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline void __stcs(ushort2 *ptr, ushort2 value) {int volatile ___ = 1;(void)ptr;(void)value;::exit(___);}
#if 0
# 397
{ } 
#endif
# 398 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline void __stcs(ushort4 *ptr, ushort4 value) {int volatile ___ = 1;(void)ptr;(void)value;::exit(___);}
#if 0
# 398
{ } 
#endif
# 399 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline void __stcs(uint2 *ptr, uint2 value) {int volatile ___ = 1;(void)ptr;(void)value;::exit(___);}
#if 0
# 399
{ } 
#endif
# 400 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline void __stcs(uint4 *ptr, uint4 value) {int volatile ___ = 1;(void)ptr;(void)value;::exit(___);}
#if 0
# 400
{ } 
#endif
# 401 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline void __stcs(ulonglong2 *ptr, ulonglong2 value) {int volatile ___ = 1;(void)ptr;(void)value;::exit(___);}
#if 0
# 401
{ } 
#endif
# 403 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline void __stcs(float *ptr, float value) {int volatile ___ = 1;(void)ptr;(void)value;::exit(___);}
#if 0
# 403
{ } 
#endif
# 404 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline void __stcs(double *ptr, double value) {int volatile ___ = 1;(void)ptr;(void)value;::exit(___);}
#if 0
# 404
{ } 
#endif
# 405 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline void __stcs(float2 *ptr, float2 value) {int volatile ___ = 1;(void)ptr;(void)value;::exit(___);}
#if 0
# 405
{ } 
#endif
# 406 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline void __stcs(float4 *ptr, float4 value) {int volatile ___ = 1;(void)ptr;(void)value;::exit(___);}
#if 0
# 406
{ } 
#endif
# 407 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline void __stcs(double2 *ptr, double2 value) {int volatile ___ = 1;(void)ptr;(void)value;::exit(___);}
#if 0
# 407
{ } 
#endif
# 411 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline void __stwt(long *ptr, long value) {int volatile ___ = 1;(void)ptr;(void)value;::exit(___);}
#if 0
# 411
{ } 
#endif
# 412 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline void __stwt(unsigned long *ptr, unsigned long value) {int volatile ___ = 1;(void)ptr;(void)value;::exit(___);}
#if 0
# 412
{ } 
#endif
# 414 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline void __stwt(char *ptr, char value) {int volatile ___ = 1;(void)ptr;(void)value;::exit(___);}
#if 0
# 414
{ } 
#endif
# 415 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline void __stwt(signed char *ptr, signed char value) {int volatile ___ = 1;(void)ptr;(void)value;::exit(___);}
#if 0
# 415
{ } 
#endif
# 416 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline void __stwt(short *ptr, short value) {int volatile ___ = 1;(void)ptr;(void)value;::exit(___);}
#if 0
# 416
{ } 
#endif
# 417 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline void __stwt(int *ptr, int value) {int volatile ___ = 1;(void)ptr;(void)value;::exit(___);}
#if 0
# 417
{ } 
#endif
# 418 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline void __stwt(long long *ptr, long long value) {int volatile ___ = 1;(void)ptr;(void)value;::exit(___);}
#if 0
# 418
{ } 
#endif
# 419 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline void __stwt(char2 *ptr, char2 value) {int volatile ___ = 1;(void)ptr;(void)value;::exit(___);}
#if 0
# 419
{ } 
#endif
# 420 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline void __stwt(char4 *ptr, char4 value) {int volatile ___ = 1;(void)ptr;(void)value;::exit(___);}
#if 0
# 420
{ } 
#endif
# 421 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline void __stwt(short2 *ptr, short2 value) {int volatile ___ = 1;(void)ptr;(void)value;::exit(___);}
#if 0
# 421
{ } 
#endif
# 422 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline void __stwt(short4 *ptr, short4 value) {int volatile ___ = 1;(void)ptr;(void)value;::exit(___);}
#if 0
# 422
{ } 
#endif
# 423 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline void __stwt(int2 *ptr, int2 value) {int volatile ___ = 1;(void)ptr;(void)value;::exit(___);}
#if 0
# 423
{ } 
#endif
# 424 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline void __stwt(int4 *ptr, int4 value) {int volatile ___ = 1;(void)ptr;(void)value;::exit(___);}
#if 0
# 424
{ } 
#endif
# 425 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline void __stwt(longlong2 *ptr, longlong2 value) {int volatile ___ = 1;(void)ptr;(void)value;::exit(___);}
#if 0
# 425
{ } 
#endif
# 427 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline void __stwt(unsigned char *ptr, unsigned char value) {int volatile ___ = 1;(void)ptr;(void)value;::exit(___);}
#if 0
# 427
{ } 
#endif
# 428 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline void __stwt(unsigned short *ptr, unsigned short value) {int volatile ___ = 1;(void)ptr;(void)value;::exit(___);}
#if 0
# 428
{ } 
#endif
# 429 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline void __stwt(unsigned *ptr, unsigned value) {int volatile ___ = 1;(void)ptr;(void)value;::exit(___);}
#if 0
# 429
{ } 
#endif
# 430 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline void __stwt(unsigned long long *ptr, unsigned long long value) {int volatile ___ = 1;(void)ptr;(void)value;::exit(___);}
#if 0
# 430
{ } 
#endif
# 431 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline void __stwt(uchar2 *ptr, uchar2 value) {int volatile ___ = 1;(void)ptr;(void)value;::exit(___);}
#if 0
# 431
{ } 
#endif
# 432 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline void __stwt(uchar4 *ptr, uchar4 value) {int volatile ___ = 1;(void)ptr;(void)value;::exit(___);}
#if 0
# 432
{ } 
#endif
# 433 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline void __stwt(ushort2 *ptr, ushort2 value) {int volatile ___ = 1;(void)ptr;(void)value;::exit(___);}
#if 0
# 433
{ } 
#endif
# 434 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline void __stwt(ushort4 *ptr, ushort4 value) {int volatile ___ = 1;(void)ptr;(void)value;::exit(___);}
#if 0
# 434
{ } 
#endif
# 435 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline void __stwt(uint2 *ptr, uint2 value) {int volatile ___ = 1;(void)ptr;(void)value;::exit(___);}
#if 0
# 435
{ } 
#endif
# 436 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline void __stwt(uint4 *ptr, uint4 value) {int volatile ___ = 1;(void)ptr;(void)value;::exit(___);}
#if 0
# 436
{ } 
#endif
# 437 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline void __stwt(ulonglong2 *ptr, ulonglong2 value) {int volatile ___ = 1;(void)ptr;(void)value;::exit(___);}
#if 0
# 437
{ } 
#endif
# 439 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline void __stwt(float *ptr, float value) {int volatile ___ = 1;(void)ptr;(void)value;::exit(___);}
#if 0
# 439
{ } 
#endif
# 440 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline void __stwt(double *ptr, double value) {int volatile ___ = 1;(void)ptr;(void)value;::exit(___);}
#if 0
# 440
{ } 
#endif
# 441 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline void __stwt(float2 *ptr, float2 value) {int volatile ___ = 1;(void)ptr;(void)value;::exit(___);}
#if 0
# 441
{ } 
#endif
# 442 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline void __stwt(float4 *ptr, float4 value) {int volatile ___ = 1;(void)ptr;(void)value;::exit(___);}
#if 0
# 442
{ } 
#endif
# 443 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline void __stwt(double2 *ptr, double2 value) {int volatile ___ = 1;(void)ptr;(void)value;::exit(___);}
#if 0
# 443
{ } 
#endif
# 460 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline unsigned __funnelshift_l(unsigned lo, unsigned hi, unsigned shift) {int volatile ___ = 1;(void)lo;(void)hi;(void)shift;::exit(___);}
#if 0
# 460
{ } 
#endif
# 472 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline unsigned __funnelshift_lc(unsigned lo, unsigned hi, unsigned shift) {int volatile ___ = 1;(void)lo;(void)hi;(void)shift;::exit(___);}
#if 0
# 472
{ } 
#endif
# 485 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline unsigned __funnelshift_r(unsigned lo, unsigned hi, unsigned shift) {int volatile ___ = 1;(void)lo;(void)hi;(void)shift;::exit(___);}
#if 0
# 485
{ } 
#endif
# 497 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline unsigned __funnelshift_rc(unsigned lo, unsigned hi, unsigned shift) {int volatile ___ = 1;(void)lo;(void)hi;(void)shift;::exit(___);}
#if 0
# 497
{ } 
#endif
# 89 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/sm_61_intrinsics.h"
__attribute__((unused)) static inline int __dp2a_lo(int srcA, int srcB, int c) {int volatile ___ = 1;(void)srcA;(void)srcB;(void)c;::exit(___);}
#if 0
# 89
{ } 
#endif
# 90 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/sm_61_intrinsics.h"
__attribute__((unused)) static inline unsigned __dp2a_lo(unsigned srcA, unsigned srcB, unsigned c) {int volatile ___ = 1;(void)srcA;(void)srcB;(void)c;::exit(___);}
#if 0
# 90
{ } 
#endif
# 92 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/sm_61_intrinsics.h"
__attribute__((unused)) static inline int __dp2a_lo(short2 srcA, char4 srcB, int c) {int volatile ___ = 1;(void)srcA;(void)srcB;(void)c;::exit(___);}
#if 0
# 92
{ } 
#endif
# 93 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/sm_61_intrinsics.h"
__attribute__((unused)) static inline unsigned __dp2a_lo(ushort2 srcA, uchar4 srcB, unsigned c) {int volatile ___ = 1;(void)srcA;(void)srcB;(void)c;::exit(___);}
#if 0
# 93
{ } 
#endif
# 95 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/sm_61_intrinsics.h"
__attribute__((unused)) static inline int __dp2a_hi(int srcA, int srcB, int c) {int volatile ___ = 1;(void)srcA;(void)srcB;(void)c;::exit(___);}
#if 0
# 95
{ } 
#endif
# 96 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/sm_61_intrinsics.h"
__attribute__((unused)) static inline unsigned __dp2a_hi(unsigned srcA, unsigned srcB, unsigned c) {int volatile ___ = 1;(void)srcA;(void)srcB;(void)c;::exit(___);}
#if 0
# 96
{ } 
#endif
# 98 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/sm_61_intrinsics.h"
__attribute__((unused)) static inline int __dp2a_hi(short2 srcA, char4 srcB, int c) {int volatile ___ = 1;(void)srcA;(void)srcB;(void)c;::exit(___);}
#if 0
# 98
{ } 
#endif
# 99 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/sm_61_intrinsics.h"
__attribute__((unused)) static inline unsigned __dp2a_hi(ushort2 srcA, uchar4 srcB, unsigned c) {int volatile ___ = 1;(void)srcA;(void)srcB;(void)c;::exit(___);}
#if 0
# 99
{ } 
#endif
# 106 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/sm_61_intrinsics.h"
__attribute__((unused)) static inline int __dp4a(int srcA, int srcB, int c) {int volatile ___ = 1;(void)srcA;(void)srcB;(void)c;::exit(___);}
#if 0
# 106
{ } 
#endif
# 107 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/sm_61_intrinsics.h"
__attribute__((unused)) static inline unsigned __dp4a(unsigned srcA, unsigned srcB, unsigned c) {int volatile ___ = 1;(void)srcA;(void)srcB;(void)c;::exit(___);}
#if 0
# 107
{ } 
#endif
# 109 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/sm_61_intrinsics.h"
__attribute__((unused)) static inline int __dp4a(char4 srcA, char4 srcB, int c) {int volatile ___ = 1;(void)srcA;(void)srcB;(void)c;::exit(___);}
#if 0
# 109
{ } 
#endif
# 110 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/sm_61_intrinsics.h"
__attribute__((unused)) static inline unsigned __dp4a(uchar4 srcA, uchar4 srcB, unsigned c) {int volatile ___ = 1;(void)srcA;(void)srcB;(void)c;::exit(___);}
#if 0
# 110
{ } 
#endif
# 98 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/crt/sm_70_rt.h"
__attribute__((unused)) static inline unsigned __match_any_sync(unsigned mask, unsigned value) {int volatile ___ = 1;(void)mask;(void)value;::exit(___);}
#if 0
# 98
{ } 
#endif
# 99 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/crt/sm_70_rt.h"
__attribute__((unused)) static inline unsigned __match_any_sync(unsigned mask, int value) {int volatile ___ = 1;(void)mask;(void)value;::exit(___);}
#if 0
# 99
{ } 
#endif
# 100 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/crt/sm_70_rt.h"
__attribute__((unused)) static inline unsigned __match_any_sync(unsigned mask, unsigned long value) {int volatile ___ = 1;(void)mask;(void)value;::exit(___);}
#if 0
# 100
{ } 
#endif
# 101 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/crt/sm_70_rt.h"
__attribute__((unused)) static inline unsigned __match_any_sync(unsigned mask, long value) {int volatile ___ = 1;(void)mask;(void)value;::exit(___);}
#if 0
# 101
{ } 
#endif
# 102 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/crt/sm_70_rt.h"
__attribute__((unused)) static inline unsigned __match_any_sync(unsigned mask, unsigned long long value) {int volatile ___ = 1;(void)mask;(void)value;::exit(___);}
#if 0
# 102
{ } 
#endif
# 103 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/crt/sm_70_rt.h"
__attribute__((unused)) static inline unsigned __match_any_sync(unsigned mask, long long value) {int volatile ___ = 1;(void)mask;(void)value;::exit(___);}
#if 0
# 103
{ } 
#endif
# 104 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/crt/sm_70_rt.h"
__attribute__((unused)) static inline unsigned __match_any_sync(unsigned mask, float value) {int volatile ___ = 1;(void)mask;(void)value;::exit(___);}
#if 0
# 104
{ } 
#endif
# 105 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/crt/sm_70_rt.h"
__attribute__((unused)) static inline unsigned __match_any_sync(unsigned mask, double value) {int volatile ___ = 1;(void)mask;(void)value;::exit(___);}
#if 0
# 105
{ } 
#endif
# 107 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/crt/sm_70_rt.h"
__attribute__((unused)) static inline unsigned __match_all_sync(unsigned mask, unsigned value, int *pred) {int volatile ___ = 1;(void)mask;(void)value;(void)pred;::exit(___);}
#if 0
# 107
{ } 
#endif
# 108 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/crt/sm_70_rt.h"
__attribute__((unused)) static inline unsigned __match_all_sync(unsigned mask, int value, int *pred) {int volatile ___ = 1;(void)mask;(void)value;(void)pred;::exit(___);}
#if 0
# 108
{ } 
#endif
# 109 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/crt/sm_70_rt.h"
__attribute__((unused)) static inline unsigned __match_all_sync(unsigned mask, unsigned long value, int *pred) {int volatile ___ = 1;(void)mask;(void)value;(void)pred;::exit(___);}
#if 0
# 109
{ } 
#endif
# 110 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/crt/sm_70_rt.h"
__attribute__((unused)) static inline unsigned __match_all_sync(unsigned mask, long value, int *pred) {int volatile ___ = 1;(void)mask;(void)value;(void)pred;::exit(___);}
#if 0
# 110
{ } 
#endif
# 111 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/crt/sm_70_rt.h"
__attribute__((unused)) static inline unsigned __match_all_sync(unsigned mask, unsigned long long value, int *pred) {int volatile ___ = 1;(void)mask;(void)value;(void)pred;::exit(___);}
#if 0
# 111
{ } 
#endif
# 112 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/crt/sm_70_rt.h"
__attribute__((unused)) static inline unsigned __match_all_sync(unsigned mask, long long value, int *pred) {int volatile ___ = 1;(void)mask;(void)value;(void)pred;::exit(___);}
#if 0
# 112
{ } 
#endif
# 113 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/crt/sm_70_rt.h"
__attribute__((unused)) static inline unsigned __match_all_sync(unsigned mask, float value, int *pred) {int volatile ___ = 1;(void)mask;(void)value;(void)pred;::exit(___);}
#if 0
# 113
{ } 
#endif
# 114 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/crt/sm_70_rt.h"
__attribute__((unused)) static inline unsigned __match_all_sync(unsigned mask, double value, int *pred) {int volatile ___ = 1;(void)mask;(void)value;(void)pred;::exit(___);}
#if 0
# 114
{ } 
#endif
# 116 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/crt/sm_70_rt.h"
__attribute__((unused)) static inline void __nanosleep(unsigned ns) {int volatile ___ = 1;(void)ns;::exit(___);}
#if 0
# 116
{ } 
#endif
# 118 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/crt/sm_70_rt.h"
__attribute__((unused)) static inline unsigned short atomicCAS(unsigned short *address, unsigned short compare, unsigned short val) {int volatile ___ = 1;(void)address;(void)compare;(void)val;::exit(___);}
#if 0
# 118
{ } 
#endif
# 98 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/crt/sm_80_rt.h"
__attribute__((unused)) static inline unsigned __reduce_add_sync(unsigned mask, unsigned value) {int volatile ___ = 1;(void)mask;(void)value;::exit(___);}
#if 0
# 98
{ } 
#endif
# 99 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/crt/sm_80_rt.h"
__attribute__((unused)) static inline unsigned __reduce_min_sync(unsigned mask, unsigned value) {int volatile ___ = 1;(void)mask;(void)value;::exit(___);}
#if 0
# 99
{ } 
#endif
# 100 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/crt/sm_80_rt.h"
__attribute__((unused)) static inline unsigned __reduce_max_sync(unsigned mask, unsigned value) {int volatile ___ = 1;(void)mask;(void)value;::exit(___);}
#if 0
# 100
{ } 
#endif
# 102 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/crt/sm_80_rt.h"
__attribute__((unused)) static inline int __reduce_add_sync(unsigned mask, int value) {int volatile ___ = 1;(void)mask;(void)value;::exit(___);}
#if 0
# 102
{ } 
#endif
# 103 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/crt/sm_80_rt.h"
__attribute__((unused)) static inline int __reduce_min_sync(unsigned mask, int value) {int volatile ___ = 1;(void)mask;(void)value;::exit(___);}
#if 0
# 103
{ } 
#endif
# 104 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/crt/sm_80_rt.h"
__attribute__((unused)) static inline int __reduce_max_sync(unsigned mask, int value) {int volatile ___ = 1;(void)mask;(void)value;::exit(___);}
#if 0
# 104
{ } 
#endif
# 106 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/crt/sm_80_rt.h"
__attribute__((unused)) static inline unsigned __reduce_and_sync(unsigned mask, unsigned value) {int volatile ___ = 1;(void)mask;(void)value;::exit(___);}
#if 0
# 106
{ } 
#endif
# 107 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/crt/sm_80_rt.h"
__attribute__((unused)) static inline unsigned __reduce_or_sync(unsigned mask, unsigned value) {int volatile ___ = 1;(void)mask;(void)value;::exit(___);}
#if 0
# 107
{ } 
#endif
# 108 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/crt/sm_80_rt.h"
__attribute__((unused)) static inline unsigned __reduce_xor_sync(unsigned mask, unsigned value) {int volatile ___ = 1;(void)mask;(void)value;::exit(___);}
#if 0
# 108
{ } 
#endif
# 111 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/crt/sm_80_rt.h"
extern "C" {
# 112
__attribute__((unused)) inline void *__nv_associate_access_property(const void *ptr, unsigned 
# 113
policy) {int volatile ___ = 1;(void)ptr;(void)policy;::exit(___);}
#if 0
# 113
{ } 
#endif
# 114 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/crt/sm_80_rt.h"
__attribute__((unused)) inline void __nv_memcpy_async_shared_global_4(void *dst, const void *src, unsigned 
# 115
src_size) {int volatile ___ = 1;(void)dst;(void)src;(void)src_size;::exit(___);}
#if 0
# 115
{ } 
#endif
# 116 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/crt/sm_80_rt.h"
__attribute__((unused)) inline void __nv_memcpy_async_shared_global_8(void *dst, const void *src, unsigned 
# 117
src_size) {int volatile ___ = 1;(void)dst;(void)src;(void)src_size;::exit(___);}
#if 0
# 117
{ } 
#endif
# 118 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/crt/sm_80_rt.h"
__attribute__((unused)) inline void __nv_memcpy_async_shared_global_16(void *dst, const void *src, unsigned 
# 119
src_size) {int volatile ___ = 1;(void)dst;(void)src;(void)src_size;::exit(___);}
#if 0
# 119
{ } 
#endif
# 120 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/crt/sm_80_rt.h"
}
# 122 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/surface_functions.h"
template< class T> 
# 123
__attribute((deprecated)) __attribute((always_inline)) __attribute__((unused)) static inline void surf1Dread(T *res, surface< void, 1>  surf, int x, int s, cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap) 
# 124
{int volatile ___ = 1;(void)res;(void)surf;(void)x;(void)s;(void)mode;
# 128
::exit(___);}
#if 0
# 124
{ 
# 128
} 
#endif
# 130 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/surface_functions.h"
template< class T> 
# 131
__attribute((deprecated)) __attribute((always_inline)) __attribute__((unused)) static inline T surf1Dread(surface< void, 1>  surf, int x, cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap) 
# 132
{int volatile ___ = 1;(void)surf;(void)x;(void)mode;
# 138
::exit(___);}
#if 0
# 132
{ 
# 138
} 
#endif
# 140 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/surface_functions.h"
template< class T> 
# 141
__attribute((deprecated)) __attribute((always_inline)) __attribute__((unused)) static inline void surf1Dread(T *res, surface< void, 1>  surf, int x, cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap) 
# 142
{int volatile ___ = 1;(void)res;(void)surf;(void)x;(void)mode;
# 146
::exit(___);}
#if 0
# 142
{ 
# 146
} 
#endif
# 149 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/surface_functions.h"
template< class T> 
# 150
__attribute((deprecated)) __attribute((always_inline)) __attribute__((unused)) static inline void surf2Dread(T *res, surface< void, 2>  surf, int x, int y, int s, cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap) 
# 151
{int volatile ___ = 1;(void)res;(void)surf;(void)x;(void)y;(void)s;(void)mode;
# 155
::exit(___);}
#if 0
# 151
{ 
# 155
} 
#endif
# 157 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/surface_functions.h"
template< class T> 
# 158
__attribute((deprecated)) __attribute((always_inline)) __attribute__((unused)) static inline T surf2Dread(surface< void, 2>  surf, int x, int y, cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap) 
# 159
{int volatile ___ = 1;(void)surf;(void)x;(void)y;(void)mode;
# 165
::exit(___);}
#if 0
# 159
{ 
# 165
} 
#endif
# 167 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/surface_functions.h"
template< class T> 
# 168
__attribute((deprecated)) __attribute((always_inline)) __attribute__((unused)) static inline void surf2Dread(T *res, surface< void, 2>  surf, int x, int y, cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap) 
# 169
{int volatile ___ = 1;(void)res;(void)surf;(void)x;(void)y;(void)mode;
# 173
::exit(___);}
#if 0
# 169
{ 
# 173
} 
#endif
# 176 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/surface_functions.h"
template< class T> 
# 177
__attribute((deprecated)) __attribute((always_inline)) __attribute__((unused)) static inline void surf3Dread(T *res, surface< void, 3>  surf, int x, int y, int z, int s, cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap) 
# 178
{int volatile ___ = 1;(void)res;(void)surf;(void)x;(void)y;(void)z;(void)s;(void)mode;
# 182
::exit(___);}
#if 0
# 178
{ 
# 182
} 
#endif
# 184 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/surface_functions.h"
template< class T> 
# 185
__attribute((deprecated)) __attribute((always_inline)) __attribute__((unused)) static inline T surf3Dread(surface< void, 3>  surf, int x, int y, int z, cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap) 
# 186
{int volatile ___ = 1;(void)surf;(void)x;(void)y;(void)z;(void)mode;
# 192
::exit(___);}
#if 0
# 186
{ 
# 192
} 
#endif
# 194 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/surface_functions.h"
template< class T> 
# 195
__attribute((deprecated)) __attribute((always_inline)) __attribute__((unused)) static inline void surf3Dread(T *res, surface< void, 3>  surf, int x, int y, int z, cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap) 
# 196
{int volatile ___ = 1;(void)res;(void)surf;(void)x;(void)y;(void)z;(void)mode;
# 200
::exit(___);}
#if 0
# 196
{ 
# 200
} 
#endif
# 204 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/surface_functions.h"
template< class T> 
# 205
__attribute((deprecated)) __attribute((always_inline)) __attribute__((unused)) static inline void surf1DLayeredread(T *res, surface< void, 241>  surf, int x, int layer, int s, cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap) 
# 206
{int volatile ___ = 1;(void)res;(void)surf;(void)x;(void)layer;(void)s;(void)mode;
# 210
::exit(___);}
#if 0
# 206
{ 
# 210
} 
#endif
# 212 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/surface_functions.h"
template< class T> 
# 213
__attribute((deprecated)) __attribute((always_inline)) __attribute__((unused)) static inline T surf1DLayeredread(surface< void, 241>  surf, int x, int layer, cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap) 
# 214
{int volatile ___ = 1;(void)surf;(void)x;(void)layer;(void)mode;
# 220
::exit(___);}
#if 0
# 214
{ 
# 220
} 
#endif
# 223 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/surface_functions.h"
template< class T> 
# 224
__attribute((deprecated)) __attribute((always_inline)) __attribute__((unused)) static inline void surf1DLayeredread(T *res, surface< void, 241>  surf, int x, int layer, cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap) 
# 225
{int volatile ___ = 1;(void)res;(void)surf;(void)x;(void)layer;(void)mode;
# 229
::exit(___);}
#if 0
# 225
{ 
# 229
} 
#endif
# 232 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/surface_functions.h"
template< class T> 
# 233
__attribute((deprecated)) __attribute((always_inline)) __attribute__((unused)) static inline void surf2DLayeredread(T *res, surface< void, 242>  surf, int x, int y, int layer, int s, cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap) 
# 234
{int volatile ___ = 1;(void)res;(void)surf;(void)x;(void)y;(void)layer;(void)s;(void)mode;
# 238
::exit(___);}
#if 0
# 234
{ 
# 238
} 
#endif
# 240 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/surface_functions.h"
template< class T> 
# 241
__attribute((deprecated)) __attribute((always_inline)) __attribute__((unused)) static inline T surf2DLayeredread(surface< void, 242>  surf, int x, int y, int layer, cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap) 
# 242
{int volatile ___ = 1;(void)surf;(void)x;(void)y;(void)layer;(void)mode;
# 248
::exit(___);}
#if 0
# 242
{ 
# 248
} 
#endif
# 251 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/surface_functions.h"
template< class T> 
# 252
__attribute((deprecated)) __attribute((always_inline)) __attribute__((unused)) static inline void surf2DLayeredread(T *res, surface< void, 242>  surf, int x, int y, int layer, cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap) 
# 253
{int volatile ___ = 1;(void)res;(void)surf;(void)x;(void)y;(void)layer;(void)mode;
# 257
::exit(___);}
#if 0
# 253
{ 
# 257
} 
#endif
# 260 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/surface_functions.h"
template< class T> 
# 261
__attribute((always_inline)) __attribute__((unused)) static inline void surfCubemapread(T *res, surface< void, 12>  surf, int x, int y, int face, int s, cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap) 
# 262
{int volatile ___ = 1;(void)res;(void)surf;(void)x;(void)y;(void)face;(void)s;(void)mode;
# 266
::exit(___);}
#if 0
# 262
{ 
# 266
} 
#endif
# 268 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/surface_functions.h"
template< class T> 
# 269
__attribute((deprecated)) __attribute((always_inline)) __attribute__((unused)) static inline T surfCubemapread(surface< void, 12>  surf, int x, int y, int face, cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap) 
# 270
{int volatile ___ = 1;(void)surf;(void)x;(void)y;(void)face;(void)mode;
# 277
::exit(___);}
#if 0
# 270
{ 
# 277
} 
#endif
# 279 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/surface_functions.h"
template< class T> 
# 280
__attribute((deprecated)) __attribute((always_inline)) __attribute__((unused)) static inline void surfCubemapread(T *res, surface< void, 12>  surf, int x, int y, int face, cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap) 
# 281
{int volatile ___ = 1;(void)res;(void)surf;(void)x;(void)y;(void)face;(void)mode;
# 285
::exit(___);}
#if 0
# 281
{ 
# 285
} 
#endif
# 288 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/surface_functions.h"
template< class T> 
# 289
__attribute((deprecated)) __attribute((always_inline)) __attribute__((unused)) static inline void surfCubemapLayeredread(T *res, surface< void, 252>  surf, int x, int y, int layerFace, int s, cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap) 
# 290
{int volatile ___ = 1;(void)res;(void)surf;(void)x;(void)y;(void)layerFace;(void)s;(void)mode;
# 294
::exit(___);}
#if 0
# 290
{ 
# 294
} 
#endif
# 296 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/surface_functions.h"
template< class T> 
# 297
__attribute((deprecated)) __attribute((always_inline)) __attribute__((unused)) static inline T surfCubemapLayeredread(surface< void, 252>  surf, int x, int y, int layerFace, cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap) 
# 298
{int volatile ___ = 1;(void)surf;(void)x;(void)y;(void)layerFace;(void)mode;
# 304
::exit(___);}
#if 0
# 298
{ 
# 304
} 
#endif
# 306 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/surface_functions.h"
template< class T> 
# 307
__attribute((deprecated)) __attribute((always_inline)) __attribute__((unused)) static inline void surfCubemapLayeredread(T *res, surface< void, 252>  surf, int x, int y, int layerFace, cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap) 
# 308
{int volatile ___ = 1;(void)res;(void)surf;(void)x;(void)y;(void)layerFace;(void)mode;
# 312
::exit(___);}
#if 0
# 308
{ 
# 312
} 
#endif
# 315 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/surface_functions.h"
template< class T> 
# 316
__attribute((deprecated)) __attribute((always_inline)) __attribute__((unused)) static inline void surf1Dwrite(T val, surface< void, 1>  surf, int x, int s, cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap) 
# 317
{int volatile ___ = 1;(void)val;(void)surf;(void)x;(void)s;(void)mode;
# 321
::exit(___);}
#if 0
# 317
{ 
# 321
} 
#endif
# 323 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/surface_functions.h"
template< class T> 
# 324
__attribute((deprecated)) __attribute((always_inline)) __attribute__((unused)) static inline void surf1Dwrite(T val, surface< void, 1>  surf, int x, cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap) 
# 325
{int volatile ___ = 1;(void)val;(void)surf;(void)x;(void)mode;
# 329
::exit(___);}
#if 0
# 325
{ 
# 329
} 
#endif
# 333 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/surface_functions.h"
template< class T> 
# 334
__attribute((deprecated)) __attribute((always_inline)) __attribute__((unused)) static inline void surf2Dwrite(T val, surface< void, 2>  surf, int x, int y, int s, cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap) 
# 335
{int volatile ___ = 1;(void)val;(void)surf;(void)x;(void)y;(void)s;(void)mode;
# 339
::exit(___);}
#if 0
# 335
{ 
# 339
} 
#endif
# 341 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/surface_functions.h"
template< class T> 
# 342
__attribute((deprecated)) __attribute((always_inline)) __attribute__((unused)) static inline void surf2Dwrite(T val, surface< void, 2>  surf, int x, int y, cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap) 
# 343
{int volatile ___ = 1;(void)val;(void)surf;(void)x;(void)y;(void)mode;
# 347
::exit(___);}
#if 0
# 343
{ 
# 347
} 
#endif
# 350 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/surface_functions.h"
template< class T> 
# 351
__attribute((deprecated)) __attribute((always_inline)) __attribute__((unused)) static inline void surf3Dwrite(T val, surface< void, 3>  surf, int x, int y, int z, int s, cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap) 
# 352
{int volatile ___ = 1;(void)val;(void)surf;(void)x;(void)y;(void)z;(void)s;(void)mode;
# 356
::exit(___);}
#if 0
# 352
{ 
# 356
} 
#endif
# 358 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/surface_functions.h"
template< class T> 
# 359
__attribute((deprecated)) __attribute((always_inline)) __attribute__((unused)) static inline void surf3Dwrite(T val, surface< void, 3>  surf, int x, int y, int z, cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap) 
# 360
{int volatile ___ = 1;(void)val;(void)surf;(void)x;(void)y;(void)z;(void)mode;
# 364
::exit(___);}
#if 0
# 360
{ 
# 364
} 
#endif
# 367 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/surface_functions.h"
template< class T> 
# 368
__attribute((deprecated)) __attribute((always_inline)) __attribute__((unused)) static inline void surf1DLayeredwrite(T val, surface< void, 241>  surf, int x, int layer, int s, cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap) 
# 369
{int volatile ___ = 1;(void)val;(void)surf;(void)x;(void)layer;(void)s;(void)mode;
# 373
::exit(___);}
#if 0
# 369
{ 
# 373
} 
#endif
# 375 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/surface_functions.h"
template< class T> 
# 376
__attribute((deprecated)) __attribute((always_inline)) __attribute__((unused)) static inline void surf1DLayeredwrite(T val, surface< void, 241>  surf, int x, int layer, cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap) 
# 377
{int volatile ___ = 1;(void)val;(void)surf;(void)x;(void)layer;(void)mode;
# 381
::exit(___);}
#if 0
# 377
{ 
# 381
} 
#endif
# 384 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/surface_functions.h"
template< class T> 
# 385
__attribute((deprecated)) __attribute((always_inline)) __attribute__((unused)) static inline void surf2DLayeredwrite(T val, surface< void, 242>  surf, int x, int y, int layer, int s, cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap) 
# 386
{int volatile ___ = 1;(void)val;(void)surf;(void)x;(void)y;(void)layer;(void)s;(void)mode;
# 390
::exit(___);}
#if 0
# 386
{ 
# 390
} 
#endif
# 392 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/surface_functions.h"
template< class T> 
# 393
__attribute((deprecated)) __attribute((always_inline)) __attribute__((unused)) static inline void surf2DLayeredwrite(T val, surface< void, 242>  surf, int x, int y, int layer, cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap) 
# 394
{int volatile ___ = 1;(void)val;(void)surf;(void)x;(void)y;(void)layer;(void)mode;
# 398
::exit(___);}
#if 0
# 394
{ 
# 398
} 
#endif
# 401 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/surface_functions.h"
template< class T> 
# 402
__attribute((deprecated)) __attribute((always_inline)) __attribute__((unused)) static inline void surfCubemapwrite(T val, surface< void, 12>  surf, int x, int y, int face, int s, cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap) 
# 403
{int volatile ___ = 1;(void)val;(void)surf;(void)x;(void)y;(void)face;(void)s;(void)mode;
# 407
::exit(___);}
#if 0
# 403
{ 
# 407
} 
#endif
# 409 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/surface_functions.h"
template< class T> 
# 410
__attribute((deprecated)) __attribute((always_inline)) __attribute__((unused)) static inline void surfCubemapwrite(T val, surface< void, 12>  surf, int x, int y, int face, cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap) 
# 411
{int volatile ___ = 1;(void)val;(void)surf;(void)x;(void)y;(void)face;(void)mode;
# 415
::exit(___);}
#if 0
# 411
{ 
# 415
} 
#endif
# 419 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/surface_functions.h"
template< class T> 
# 420
__attribute((deprecated)) __attribute((always_inline)) __attribute__((unused)) static inline void surfCubemapLayeredwrite(T val, surface< void, 252>  surf, int x, int y, int layerFace, int s, cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap) 
# 421
{int volatile ___ = 1;(void)val;(void)surf;(void)x;(void)y;(void)layerFace;(void)s;(void)mode;
# 425
::exit(___);}
#if 0
# 421
{ 
# 425
} 
#endif
# 427 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/surface_functions.h"
template< class T> 
# 428
__attribute((deprecated)) __attribute((always_inline)) __attribute__((unused)) static inline void surfCubemapLayeredwrite(T val, surface< void, 252>  surf, int x, int y, int layerFace, cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap) 
# 429
{int volatile ___ = 1;(void)val;(void)surf;(void)x;(void)y;(void)layerFace;(void)mode;
# 433
::exit(___);}
#if 0
# 429
{ 
# 433
} 
#endif
# 72 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/texture_fetch_functions.h"
template< class T> 
# 73
struct __nv_tex_rmet_ret { }; 
# 75
template<> struct __nv_tex_rmet_ret< char>  { typedef char type; }; 
# 76
template<> struct __nv_tex_rmet_ret< signed char>  { typedef signed char type; }; 
# 77
template<> struct __nv_tex_rmet_ret< unsigned char>  { typedef unsigned char type; }; 
# 78
template<> struct __nv_tex_rmet_ret< char1>  { typedef char1 type; }; 
# 79
template<> struct __nv_tex_rmet_ret< uchar1>  { typedef uchar1 type; }; 
# 80
template<> struct __nv_tex_rmet_ret< char2>  { typedef char2 type; }; 
# 81
template<> struct __nv_tex_rmet_ret< uchar2>  { typedef uchar2 type; }; 
# 82
template<> struct __nv_tex_rmet_ret< char4>  { typedef char4 type; }; 
# 83
template<> struct __nv_tex_rmet_ret< uchar4>  { typedef uchar4 type; }; 
# 85
template<> struct __nv_tex_rmet_ret< short>  { typedef short type; }; 
# 86
template<> struct __nv_tex_rmet_ret< unsigned short>  { typedef unsigned short type; }; 
# 87
template<> struct __nv_tex_rmet_ret< short1>  { typedef short1 type; }; 
# 88
template<> struct __nv_tex_rmet_ret< ushort1>  { typedef ushort1 type; }; 
# 89
template<> struct __nv_tex_rmet_ret< short2>  { typedef short2 type; }; 
# 90
template<> struct __nv_tex_rmet_ret< ushort2>  { typedef ushort2 type; }; 
# 91
template<> struct __nv_tex_rmet_ret< short4>  { typedef short4 type; }; 
# 92
template<> struct __nv_tex_rmet_ret< ushort4>  { typedef ushort4 type; }; 
# 94
template<> struct __nv_tex_rmet_ret< int>  { typedef int type; }; 
# 95
template<> struct __nv_tex_rmet_ret< unsigned>  { typedef unsigned type; }; 
# 96
template<> struct __nv_tex_rmet_ret< int1>  { typedef int1 type; }; 
# 97
template<> struct __nv_tex_rmet_ret< uint1>  { typedef uint1 type; }; 
# 98
template<> struct __nv_tex_rmet_ret< int2>  { typedef int2 type; }; 
# 99
template<> struct __nv_tex_rmet_ret< uint2>  { typedef uint2 type; }; 
# 100
template<> struct __nv_tex_rmet_ret< int4>  { typedef int4 type; }; 
# 101
template<> struct __nv_tex_rmet_ret< uint4>  { typedef uint4 type; }; 
# 113 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/texture_fetch_functions.h"
template<> struct __nv_tex_rmet_ret< float>  { typedef float type; }; 
# 114
template<> struct __nv_tex_rmet_ret< float1>  { typedef float1 type; }; 
# 115
template<> struct __nv_tex_rmet_ret< float2>  { typedef float2 type; }; 
# 116
template<> struct __nv_tex_rmet_ret< float4>  { typedef float4 type; }; 
# 119
template< class T> struct __nv_tex_rmet_cast { typedef T *type; }; 
# 131 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/texture_fetch_functions.h"
template< class T> 
# 132
__attribute((deprecated)) __attribute((always_inline)) __attribute__((unused)) static inline typename __nv_tex_rmet_ret< T> ::type tex1Dfetch(texture< T, 1, cudaReadModeElementType>  t, int x) 
# 133
{int volatile ___ = 1;(void)t;(void)x;
# 139
::exit(___);}
#if 0
# 133
{ 
# 139
} 
#endif
# 141 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/texture_fetch_functions.h"
template< class T> 
# 142
struct __nv_tex_rmnf_ret { }; 
# 144
template<> struct __nv_tex_rmnf_ret< char>  { typedef float type; }; 
# 145
template<> struct __nv_tex_rmnf_ret< signed char>  { typedef float type; }; 
# 146
template<> struct __nv_tex_rmnf_ret< unsigned char>  { typedef float type; }; 
# 147
template<> struct __nv_tex_rmnf_ret< short>  { typedef float type; }; 
# 148
template<> struct __nv_tex_rmnf_ret< unsigned short>  { typedef float type; }; 
# 149
template<> struct __nv_tex_rmnf_ret< char1>  { typedef float1 type; }; 
# 150
template<> struct __nv_tex_rmnf_ret< uchar1>  { typedef float1 type; }; 
# 151
template<> struct __nv_tex_rmnf_ret< short1>  { typedef float1 type; }; 
# 152
template<> struct __nv_tex_rmnf_ret< ushort1>  { typedef float1 type; }; 
# 153
template<> struct __nv_tex_rmnf_ret< char2>  { typedef float2 type; }; 
# 154
template<> struct __nv_tex_rmnf_ret< uchar2>  { typedef float2 type; }; 
# 155
template<> struct __nv_tex_rmnf_ret< short2>  { typedef float2 type; }; 
# 156
template<> struct __nv_tex_rmnf_ret< ushort2>  { typedef float2 type; }; 
# 157
template<> struct __nv_tex_rmnf_ret< char4>  { typedef float4 type; }; 
# 158
template<> struct __nv_tex_rmnf_ret< uchar4>  { typedef float4 type; }; 
# 159
template<> struct __nv_tex_rmnf_ret< short4>  { typedef float4 type; }; 
# 160
template<> struct __nv_tex_rmnf_ret< ushort4>  { typedef float4 type; }; 
# 162
template< class T> 
# 163
__attribute((deprecated)) __attribute((always_inline)) __attribute__((unused)) static inline typename __nv_tex_rmnf_ret< T> ::type tex1Dfetch(texture< T, 1, cudaReadModeNormalizedFloat>  t, int x) 
# 164
{int volatile ___ = 1;(void)t;(void)x;
# 171
::exit(___);}
#if 0
# 164
{ 
# 171
} 
#endif
# 174 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/texture_fetch_functions.h"
template< class T> 
# 175
__attribute((deprecated)) __attribute((always_inline)) __attribute__((unused)) static inline typename __nv_tex_rmet_ret< T> ::type tex1D(texture< T, 1, cudaReadModeElementType>  t, float x) 
# 176
{int volatile ___ = 1;(void)t;(void)x;
# 182
::exit(___);}
#if 0
# 176
{ 
# 182
} 
#endif
# 184 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/texture_fetch_functions.h"
template< class T> 
# 185
__attribute((deprecated)) __attribute((always_inline)) __attribute__((unused)) static inline typename __nv_tex_rmnf_ret< T> ::type tex1D(texture< T, 1, cudaReadModeNormalizedFloat>  t, float x) 
# 186
{int volatile ___ = 1;(void)t;(void)x;
# 193
::exit(___);}
#if 0
# 186
{ 
# 193
} 
#endif
# 197 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/texture_fetch_functions.h"
template< class T> 
# 198
__attribute((deprecated)) __attribute((always_inline)) __attribute__((unused)) static inline typename __nv_tex_rmet_ret< T> ::type tex2D(texture< T, 2, cudaReadModeElementType>  t, float x, float y) 
# 199
{int volatile ___ = 1;(void)t;(void)x;(void)y;
# 206
::exit(___);}
#if 0
# 199
{ 
# 206
} 
#endif
# 208 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/texture_fetch_functions.h"
template< class T> 
# 209
__attribute((deprecated)) __attribute((always_inline)) __attribute__((unused)) static inline typename __nv_tex_rmnf_ret< T> ::type tex2D(texture< T, 2, cudaReadModeNormalizedFloat>  t, float x, float y) 
# 210
{int volatile ___ = 1;(void)t;(void)x;(void)y;
# 217
::exit(___);}
#if 0
# 210
{ 
# 217
} 
#endif
# 221 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/texture_fetch_functions.h"
template< class T> 
# 222
__attribute((deprecated)) __attribute((always_inline)) __attribute__((unused)) static inline typename __nv_tex_rmet_ret< T> ::type tex1DLayered(texture< T, 241, cudaReadModeElementType>  t, float x, int layer) 
# 223
{int volatile ___ = 1;(void)t;(void)x;(void)layer;
# 229
::exit(___);}
#if 0
# 223
{ 
# 229
} 
#endif
# 231 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/texture_fetch_functions.h"
template< class T> 
# 232
__attribute((deprecated)) __attribute((always_inline)) __attribute__((unused)) static inline typename __nv_tex_rmnf_ret< T> ::type tex1DLayered(texture< T, 241, cudaReadModeNormalizedFloat>  t, float x, int layer) 
# 233
{int volatile ___ = 1;(void)t;(void)x;(void)layer;
# 240
::exit(___);}
#if 0
# 233
{ 
# 240
} 
#endif
# 244 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/texture_fetch_functions.h"
template< class T> 
# 245
__attribute((deprecated)) __attribute((always_inline)) __attribute__((unused)) static inline typename __nv_tex_rmet_ret< T> ::type tex2DLayered(texture< T, 242, cudaReadModeElementType>  t, float x, float y, int layer) 
# 246
{int volatile ___ = 1;(void)t;(void)x;(void)y;(void)layer;
# 252
::exit(___);}
#if 0
# 246
{ 
# 252
} 
#endif
# 254 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/texture_fetch_functions.h"
template< class T> 
# 255
__attribute((deprecated)) __attribute((always_inline)) __attribute__((unused)) static inline typename __nv_tex_rmnf_ret< T> ::type tex2DLayered(texture< T, 242, cudaReadModeNormalizedFloat>  t, float x, float y, int layer) 
# 256
{int volatile ___ = 1;(void)t;(void)x;(void)y;(void)layer;
# 263
::exit(___);}
#if 0
# 256
{ 
# 263
} 
#endif
# 266 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/texture_fetch_functions.h"
template< class T> 
# 267
__attribute((deprecated)) __attribute((always_inline)) __attribute__((unused)) static inline typename __nv_tex_rmet_ret< T> ::type tex3D(texture< T, 3, cudaReadModeElementType>  t, float x, float y, float z) 
# 268
{int volatile ___ = 1;(void)t;(void)x;(void)y;(void)z;
# 274
::exit(___);}
#if 0
# 268
{ 
# 274
} 
#endif
# 276 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/texture_fetch_functions.h"
template< class T> 
# 277
__attribute((deprecated)) __attribute((always_inline)) __attribute__((unused)) static inline typename __nv_tex_rmnf_ret< T> ::type tex3D(texture< T, 3, cudaReadModeNormalizedFloat>  t, float x, float y, float z) 
# 278
{int volatile ___ = 1;(void)t;(void)x;(void)y;(void)z;
# 285
::exit(___);}
#if 0
# 278
{ 
# 285
} 
#endif
# 288 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/texture_fetch_functions.h"
template< class T> 
# 289
__attribute((deprecated)) __attribute((always_inline)) __attribute__((unused)) static inline typename __nv_tex_rmet_ret< T> ::type texCubemap(texture< T, 12, cudaReadModeElementType>  t, float x, float y, float z) 
# 290
{int volatile ___ = 1;(void)t;(void)x;(void)y;(void)z;
# 296
::exit(___);}
#if 0
# 290
{ 
# 296
} 
#endif
# 298 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/texture_fetch_functions.h"
template< class T> 
# 299
__attribute((deprecated)) __attribute((always_inline)) __attribute__((unused)) static inline typename __nv_tex_rmnf_ret< T> ::type texCubemap(texture< T, 12, cudaReadModeNormalizedFloat>  t, float x, float y, float z) 
# 300
{int volatile ___ = 1;(void)t;(void)x;(void)y;(void)z;
# 307
::exit(___);}
#if 0
# 300
{ 
# 307
} 
#endif
# 310 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/texture_fetch_functions.h"
template< class T> 
# 311
struct __nv_tex2dgather_ret { }; 
# 312
template<> struct __nv_tex2dgather_ret< char>  { typedef char4 type; }; 
# 313
template<> struct __nv_tex2dgather_ret< signed char>  { typedef char4 type; }; 
# 314
template<> struct __nv_tex2dgather_ret< char1>  { typedef char4 type; }; 
# 315
template<> struct __nv_tex2dgather_ret< char2>  { typedef char4 type; }; 
# 316
template<> struct __nv_tex2dgather_ret< char3>  { typedef char4 type; }; 
# 317
template<> struct __nv_tex2dgather_ret< char4>  { typedef char4 type; }; 
# 318
template<> struct __nv_tex2dgather_ret< unsigned char>  { typedef uchar4 type; }; 
# 319
template<> struct __nv_tex2dgather_ret< uchar1>  { typedef uchar4 type; }; 
# 320
template<> struct __nv_tex2dgather_ret< uchar2>  { typedef uchar4 type; }; 
# 321
template<> struct __nv_tex2dgather_ret< uchar3>  { typedef uchar4 type; }; 
# 322
template<> struct __nv_tex2dgather_ret< uchar4>  { typedef uchar4 type; }; 
# 324
template<> struct __nv_tex2dgather_ret< short>  { typedef short4 type; }; 
# 325
template<> struct __nv_tex2dgather_ret< short1>  { typedef short4 type; }; 
# 326
template<> struct __nv_tex2dgather_ret< short2>  { typedef short4 type; }; 
# 327
template<> struct __nv_tex2dgather_ret< short3>  { typedef short4 type; }; 
# 328
template<> struct __nv_tex2dgather_ret< short4>  { typedef short4 type; }; 
# 329
template<> struct __nv_tex2dgather_ret< unsigned short>  { typedef ushort4 type; }; 
# 330
template<> struct __nv_tex2dgather_ret< ushort1>  { typedef ushort4 type; }; 
# 331
template<> struct __nv_tex2dgather_ret< ushort2>  { typedef ushort4 type; }; 
# 332
template<> struct __nv_tex2dgather_ret< ushort3>  { typedef ushort4 type; }; 
# 333
template<> struct __nv_tex2dgather_ret< ushort4>  { typedef ushort4 type; }; 
# 335
template<> struct __nv_tex2dgather_ret< int>  { typedef int4 type; }; 
# 336
template<> struct __nv_tex2dgather_ret< int1>  { typedef int4 type; }; 
# 337
template<> struct __nv_tex2dgather_ret< int2>  { typedef int4 type; }; 
# 338
template<> struct __nv_tex2dgather_ret< int3>  { typedef int4 type; }; 
# 339
template<> struct __nv_tex2dgather_ret< int4>  { typedef int4 type; }; 
# 340
template<> struct __nv_tex2dgather_ret< unsigned>  { typedef uint4 type; }; 
# 341
template<> struct __nv_tex2dgather_ret< uint1>  { typedef uint4 type; }; 
# 342
template<> struct __nv_tex2dgather_ret< uint2>  { typedef uint4 type; }; 
# 343
template<> struct __nv_tex2dgather_ret< uint3>  { typedef uint4 type; }; 
# 344
template<> struct __nv_tex2dgather_ret< uint4>  { typedef uint4 type; }; 
# 346
template<> struct __nv_tex2dgather_ret< float>  { typedef float4 type; }; 
# 347
template<> struct __nv_tex2dgather_ret< float1>  { typedef float4 type; }; 
# 348
template<> struct __nv_tex2dgather_ret< float2>  { typedef float4 type; }; 
# 349
template<> struct __nv_tex2dgather_ret< float3>  { typedef float4 type; }; 
# 350
template<> struct __nv_tex2dgather_ret< float4>  { typedef float4 type; }; 
# 352
template< class T> 
# 353
__attribute((always_inline)) __attribute__((unused)) static inline typename __nv_tex2dgather_ret< T> ::type tex2Dgather(texture< T, 2, cudaReadModeElementType>  t, float x, float y, int comp = 0) 
# 354
{int volatile ___ = 1;(void)t;(void)x;(void)y;(void)comp;
# 361
::exit(___);}
#if 0
# 354
{ 
# 361
} 
#endif
# 364 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/texture_fetch_functions.h"
template< class T> struct __nv_tex2dgather_rmnf_ret { }; 
# 365
template<> struct __nv_tex2dgather_rmnf_ret< char>  { typedef float4 type; }; 
# 366
template<> struct __nv_tex2dgather_rmnf_ret< signed char>  { typedef float4 type; }; 
# 367
template<> struct __nv_tex2dgather_rmnf_ret< unsigned char>  { typedef float4 type; }; 
# 368
template<> struct __nv_tex2dgather_rmnf_ret< char1>  { typedef float4 type; }; 
# 369
template<> struct __nv_tex2dgather_rmnf_ret< uchar1>  { typedef float4 type; }; 
# 370
template<> struct __nv_tex2dgather_rmnf_ret< char2>  { typedef float4 type; }; 
# 371
template<> struct __nv_tex2dgather_rmnf_ret< uchar2>  { typedef float4 type; }; 
# 372
template<> struct __nv_tex2dgather_rmnf_ret< char3>  { typedef float4 type; }; 
# 373
template<> struct __nv_tex2dgather_rmnf_ret< uchar3>  { typedef float4 type; }; 
# 374
template<> struct __nv_tex2dgather_rmnf_ret< char4>  { typedef float4 type; }; 
# 375
template<> struct __nv_tex2dgather_rmnf_ret< uchar4>  { typedef float4 type; }; 
# 376
template<> struct __nv_tex2dgather_rmnf_ret< signed short>  { typedef float4 type; }; 
# 377
template<> struct __nv_tex2dgather_rmnf_ret< unsigned short>  { typedef float4 type; }; 
# 378
template<> struct __nv_tex2dgather_rmnf_ret< short1>  { typedef float4 type; }; 
# 379
template<> struct __nv_tex2dgather_rmnf_ret< ushort1>  { typedef float4 type; }; 
# 380
template<> struct __nv_tex2dgather_rmnf_ret< short2>  { typedef float4 type; }; 
# 381
template<> struct __nv_tex2dgather_rmnf_ret< ushort2>  { typedef float4 type; }; 
# 382
template<> struct __nv_tex2dgather_rmnf_ret< short3>  { typedef float4 type; }; 
# 383
template<> struct __nv_tex2dgather_rmnf_ret< ushort3>  { typedef float4 type; }; 
# 384
template<> struct __nv_tex2dgather_rmnf_ret< short4>  { typedef float4 type; }; 
# 385
template<> struct __nv_tex2dgather_rmnf_ret< ushort4>  { typedef float4 type; }; 
# 387
template< class T> 
# 388
__attribute((always_inline)) __attribute__((unused)) static inline typename __nv_tex2dgather_rmnf_ret< T> ::type tex2Dgather(texture< T, 2, cudaReadModeNormalizedFloat>  t, float x, float y, int comp = 0) 
# 389
{int volatile ___ = 1;(void)t;(void)x;(void)y;(void)comp;
# 396
::exit(___);}
#if 0
# 389
{ 
# 396
} 
#endif
# 400 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/texture_fetch_functions.h"
template< class T> 
# 401
__attribute((deprecated)) __attribute((always_inline)) __attribute__((unused)) static inline typename __nv_tex_rmet_ret< T> ::type tex1DLod(texture< T, 1, cudaReadModeElementType>  t, float x, float level) 
# 402
{int volatile ___ = 1;(void)t;(void)x;(void)level;
# 408
::exit(___);}
#if 0
# 402
{ 
# 408
} 
#endif
# 410 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/texture_fetch_functions.h"
template< class T> 
# 411
__attribute((deprecated)) __attribute((always_inline)) __attribute__((unused)) static inline typename __nv_tex_rmnf_ret< T> ::type tex1DLod(texture< T, 1, cudaReadModeNormalizedFloat>  t, float x, float level) 
# 412
{int volatile ___ = 1;(void)t;(void)x;(void)level;
# 419
::exit(___);}
#if 0
# 412
{ 
# 419
} 
#endif
# 422 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/texture_fetch_functions.h"
template< class T> 
# 423
__attribute((deprecated)) __attribute((always_inline)) __attribute__((unused)) static inline typename __nv_tex_rmet_ret< T> ::type tex2DLod(texture< T, 2, cudaReadModeElementType>  t, float x, float y, float level) 
# 424
{int volatile ___ = 1;(void)t;(void)x;(void)y;(void)level;
# 430
::exit(___);}
#if 0
# 424
{ 
# 430
} 
#endif
# 432 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/texture_fetch_functions.h"
template< class T> 
# 433
__attribute((deprecated)) __attribute((always_inline)) __attribute__((unused)) static inline typename __nv_tex_rmnf_ret< T> ::type tex2DLod(texture< T, 2, cudaReadModeNormalizedFloat>  t, float x, float y, float level) 
# 434
{int volatile ___ = 1;(void)t;(void)x;(void)y;(void)level;
# 441
::exit(___);}
#if 0
# 434
{ 
# 441
} 
#endif
# 444 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/texture_fetch_functions.h"
template< class T> 
# 445
__attribute((deprecated)) __attribute((always_inline)) __attribute__((unused)) static inline typename __nv_tex_rmet_ret< T> ::type tex1DLayeredLod(texture< T, 241, cudaReadModeElementType>  t, float x, int layer, float level) 
# 446
{int volatile ___ = 1;(void)t;(void)x;(void)layer;(void)level;
# 452
::exit(___);}
#if 0
# 446
{ 
# 452
} 
#endif
# 454 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/texture_fetch_functions.h"
template< class T> 
# 455
__attribute((deprecated)) __attribute((always_inline)) __attribute__((unused)) static inline typename __nv_tex_rmnf_ret< T> ::type tex1DLayeredLod(texture< T, 241, cudaReadModeNormalizedFloat>  t, float x, int layer, float level) 
# 456
{int volatile ___ = 1;(void)t;(void)x;(void)layer;(void)level;
# 463
::exit(___);}
#if 0
# 456
{ 
# 463
} 
#endif
# 466 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/texture_fetch_functions.h"
template< class T> 
# 467
__attribute((deprecated)) __attribute((always_inline)) __attribute__((unused)) static inline typename __nv_tex_rmet_ret< T> ::type tex2DLayeredLod(texture< T, 242, cudaReadModeElementType>  t, float x, float y, int layer, float level) 
# 468
{int volatile ___ = 1;(void)t;(void)x;(void)y;(void)layer;(void)level;
# 474
::exit(___);}
#if 0
# 468
{ 
# 474
} 
#endif
# 476 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/texture_fetch_functions.h"
template< class T> 
# 477
__attribute((deprecated)) __attribute((always_inline)) __attribute__((unused)) static inline typename __nv_tex_rmnf_ret< T> ::type tex2DLayeredLod(texture< T, 242, cudaReadModeNormalizedFloat>  t, float x, float y, int layer, float level) 
# 478
{int volatile ___ = 1;(void)t;(void)x;(void)y;(void)layer;(void)level;
# 485
::exit(___);}
#if 0
# 478
{ 
# 485
} 
#endif
# 488 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/texture_fetch_functions.h"
template< class T> 
# 489
__attribute((deprecated)) __attribute((always_inline)) __attribute__((unused)) static inline typename __nv_tex_rmet_ret< T> ::type tex3DLod(texture< T, 3, cudaReadModeElementType>  t, float x, float y, float z, float level) 
# 490
{int volatile ___ = 1;(void)t;(void)x;(void)y;(void)z;(void)level;
# 496
::exit(___);}
#if 0
# 490
{ 
# 496
} 
#endif
# 498 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/texture_fetch_functions.h"
template< class T> 
# 499
__attribute((deprecated)) __attribute((always_inline)) __attribute__((unused)) static inline typename __nv_tex_rmnf_ret< T> ::type tex3DLod(texture< T, 3, cudaReadModeNormalizedFloat>  t, float x, float y, float z, float level) 
# 500
{int volatile ___ = 1;(void)t;(void)x;(void)y;(void)z;(void)level;
# 507
::exit(___);}
#if 0
# 500
{ 
# 507
} 
#endif
# 510 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/texture_fetch_functions.h"
template< class T> 
# 511
__attribute((deprecated)) __attribute((always_inline)) __attribute__((unused)) static inline typename __nv_tex_rmet_ret< T> ::type texCubemapLod(texture< T, 12, cudaReadModeElementType>  t, float x, float y, float z, float level) 
# 512
{int volatile ___ = 1;(void)t;(void)x;(void)y;(void)z;(void)level;
# 518
::exit(___);}
#if 0
# 512
{ 
# 518
} 
#endif
# 520 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/texture_fetch_functions.h"
template< class T> 
# 521
__attribute((deprecated)) __attribute((always_inline)) __attribute__((unused)) static inline typename __nv_tex_rmnf_ret< T> ::type texCubemapLod(texture< T, 12, cudaReadModeNormalizedFloat>  t, float x, float y, float z, float level) 
# 522
{int volatile ___ = 1;(void)t;(void)x;(void)y;(void)z;(void)level;
# 529
::exit(___);}
#if 0
# 522
{ 
# 529
} 
#endif
# 533 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/texture_fetch_functions.h"
template< class T> 
# 534
__attribute((deprecated)) __attribute((always_inline)) __attribute__((unused)) static inline typename __nv_tex_rmet_ret< T> ::type texCubemapLayered(texture< T, 252, cudaReadModeElementType>  t, float x, float y, float z, int layer) 
# 535
{int volatile ___ = 1;(void)t;(void)x;(void)y;(void)z;(void)layer;
# 541
::exit(___);}
#if 0
# 535
{ 
# 541
} 
#endif
# 543 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/texture_fetch_functions.h"
template< class T> 
# 544
__attribute((deprecated)) __attribute((always_inline)) __attribute__((unused)) static inline typename __nv_tex_rmnf_ret< T> ::type texCubemapLayered(texture< T, 252, cudaReadModeNormalizedFloat>  t, float x, float y, float z, int layer) 
# 545
{int volatile ___ = 1;(void)t;(void)x;(void)y;(void)z;(void)layer;
# 552
::exit(___);}
#if 0
# 545
{ 
# 552
} 
#endif
# 556 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/texture_fetch_functions.h"
template< class T> 
# 557
__attribute((deprecated)) __attribute((always_inline)) __attribute__((unused)) static inline typename __nv_tex_rmet_ret< T> ::type texCubemapLayeredLod(texture< T, 252, cudaReadModeElementType>  t, float x, float y, float z, int layer, float level) 
# 558
{int volatile ___ = 1;(void)t;(void)x;(void)y;(void)z;(void)layer;(void)level;
# 564
::exit(___);}
#if 0
# 558
{ 
# 564
} 
#endif
# 566 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/texture_fetch_functions.h"
template< class T> 
# 567
__attribute((deprecated)) __attribute((always_inline)) __attribute__((unused)) static inline typename __nv_tex_rmnf_ret< T> ::type texCubemapLayeredLod(texture< T, 252, cudaReadModeNormalizedFloat>  t, float x, float y, float z, int layer, float level) 
# 568
{int volatile ___ = 1;(void)t;(void)x;(void)y;(void)z;(void)layer;(void)level;
# 575
::exit(___);}
#if 0
# 568
{ 
# 575
} 
#endif
# 579 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/texture_fetch_functions.h"
template< class T> 
# 580
__attribute((deprecated)) __attribute((always_inline)) __attribute__((unused)) static inline typename __nv_tex_rmet_ret< T> ::type texCubemapGrad(texture< T, 12, cudaReadModeElementType>  t, float x, float y, float z, float4 dPdx, float4 dPdy) 
# 581
{int volatile ___ = 1;(void)t;(void)x;(void)y;(void)z;(void)dPdx;(void)dPdy;
# 587
::exit(___);}
#if 0
# 581
{ 
# 587
} 
#endif
# 589 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/texture_fetch_functions.h"
template< class T> 
# 590
__attribute((deprecated)) __attribute((always_inline)) __attribute__((unused)) static inline typename __nv_tex_rmnf_ret< T> ::type texCubemapGrad(texture< T, 12, cudaReadModeNormalizedFloat>  t, float x, float y, float z, float4 dPdx, float4 dPdy) 
# 591
{int volatile ___ = 1;(void)t;(void)x;(void)y;(void)z;(void)dPdx;(void)dPdy;
# 598
::exit(___);}
#if 0
# 591
{ 
# 598
} 
#endif
# 602 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/texture_fetch_functions.h"
template< class T> 
# 603
__attribute((deprecated)) __attribute((always_inline)) __attribute__((unused)) static inline typename __nv_tex_rmet_ret< T> ::type texCubemapLayeredGrad(texture< T, 252, cudaReadModeElementType>  t, float x, float y, float z, int layer, float4 dPdx, float4 dPdy) 
# 604
{int volatile ___ = 1;(void)t;(void)x;(void)y;(void)z;(void)layer;(void)dPdx;(void)dPdy;
# 610
::exit(___);}
#if 0
# 604
{ 
# 610
} 
#endif
# 612 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/texture_fetch_functions.h"
template< class T> 
# 613
__attribute((deprecated)) __attribute((always_inline)) __attribute__((unused)) static inline typename __nv_tex_rmnf_ret< T> ::type texCubemapLayeredGrad(texture< T, 252, cudaReadModeNormalizedFloat>  t, float x, float y, float z, int layer, float4 dPdx, float4 dPdy) 
# 614
{int volatile ___ = 1;(void)t;(void)x;(void)y;(void)z;(void)layer;(void)dPdx;(void)dPdy;
# 621
::exit(___);}
#if 0
# 614
{ 
# 621
} 
#endif
# 625 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/texture_fetch_functions.h"
template< class T> 
# 626
__attribute((deprecated)) __attribute((always_inline)) __attribute__((unused)) static inline typename __nv_tex_rmet_ret< T> ::type tex1DGrad(texture< T, 1, cudaReadModeElementType>  t, float x, float dPdx, float dPdy) 
# 627
{int volatile ___ = 1;(void)t;(void)x;(void)dPdx;(void)dPdy;
# 633
::exit(___);}
#if 0
# 627
{ 
# 633
} 
#endif
# 635 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/texture_fetch_functions.h"
template< class T> 
# 636
__attribute((deprecated)) __attribute((always_inline)) __attribute__((unused)) static inline typename __nv_tex_rmnf_ret< T> ::type tex1DGrad(texture< T, 1, cudaReadModeNormalizedFloat>  t, float x, float dPdx, float dPdy) 
# 637
{int volatile ___ = 1;(void)t;(void)x;(void)dPdx;(void)dPdy;
# 644
::exit(___);}
#if 0
# 637
{ 
# 644
} 
#endif
# 648 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/texture_fetch_functions.h"
template< class T> 
# 649
__attribute((deprecated)) __attribute((always_inline)) __attribute__((unused)) static inline typename __nv_tex_rmet_ret< T> ::type tex2DGrad(texture< T, 2, cudaReadModeElementType>  t, float x, float y, float2 dPdx, float2 dPdy) 
# 650
{int volatile ___ = 1;(void)t;(void)x;(void)y;(void)dPdx;(void)dPdy;
# 656
::exit(___);}
#if 0
# 650
{ 
# 656
} 
#endif
# 658 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/texture_fetch_functions.h"
template< class T> 
# 659
__attribute((deprecated)) __attribute((always_inline)) __attribute__((unused)) static inline typename __nv_tex_rmnf_ret< T> ::type tex2DGrad(texture< T, 2, cudaReadModeNormalizedFloat>  t, float x, float y, float2 dPdx, float2 dPdy) 
# 660
{int volatile ___ = 1;(void)t;(void)x;(void)y;(void)dPdx;(void)dPdy;
# 667
::exit(___);}
#if 0
# 660
{ 
# 667
} 
#endif
# 670 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/texture_fetch_functions.h"
template< class T> 
# 671
__attribute((deprecated)) __attribute((always_inline)) __attribute__((unused)) static inline typename __nv_tex_rmet_ret< T> ::type tex1DLayeredGrad(texture< T, 241, cudaReadModeElementType>  t, float x, int layer, float dPdx, float dPdy) 
# 672
{int volatile ___ = 1;(void)t;(void)x;(void)layer;(void)dPdx;(void)dPdy;
# 678
::exit(___);}
#if 0
# 672
{ 
# 678
} 
#endif
# 680 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/texture_fetch_functions.h"
template< class T> 
# 681
__attribute((deprecated)) __attribute((always_inline)) __attribute__((unused)) static inline typename __nv_tex_rmnf_ret< T> ::type tex1DLayeredGrad(texture< T, 241, cudaReadModeNormalizedFloat>  t, float x, int layer, float dPdx, float dPdy) 
# 682
{int volatile ___ = 1;(void)t;(void)x;(void)layer;(void)dPdx;(void)dPdy;
# 689
::exit(___);}
#if 0
# 682
{ 
# 689
} 
#endif
# 692 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/texture_fetch_functions.h"
template< class T> 
# 693
__attribute((deprecated)) __attribute((always_inline)) __attribute__((unused)) static inline typename __nv_tex_rmet_ret< T> ::type tex2DLayeredGrad(texture< T, 242, cudaReadModeElementType>  t, float x, float y, int layer, float2 dPdx, float2 dPdy) 
# 694
{int volatile ___ = 1;(void)t;(void)x;(void)y;(void)layer;(void)dPdx;(void)dPdy;
# 700
::exit(___);}
#if 0
# 694
{ 
# 700
} 
#endif
# 702 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/texture_fetch_functions.h"
template< class T> 
# 703
__attribute((deprecated)) __attribute((always_inline)) __attribute__((unused)) static inline typename __nv_tex_rmnf_ret< T> ::type tex2DLayeredGrad(texture< T, 242, cudaReadModeNormalizedFloat>  t, float x, float y, int layer, float2 dPdx, float2 dPdy) 
# 704
{int volatile ___ = 1;(void)t;(void)x;(void)y;(void)layer;(void)dPdx;(void)dPdy;
# 711
::exit(___);}
#if 0
# 704
{ 
# 711
} 
#endif
# 714 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/texture_fetch_functions.h"
template< class T> 
# 715
__attribute((deprecated)) __attribute((always_inline)) __attribute__((unused)) static inline typename __nv_tex_rmet_ret< T> ::type tex3DGrad(texture< T, 3, cudaReadModeElementType>  t, float x, float y, float z, float4 dPdx, float4 dPdy) 
# 716
{int volatile ___ = 1;(void)t;(void)x;(void)y;(void)z;(void)dPdx;(void)dPdy;
# 722
::exit(___);}
#if 0
# 716
{ 
# 722
} 
#endif
# 724 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/texture_fetch_functions.h"
template< class T> 
# 725
__attribute((deprecated)) __attribute((always_inline)) __attribute__((unused)) static inline typename __nv_tex_rmnf_ret< T> ::type tex3DGrad(texture< T, 3, cudaReadModeNormalizedFloat>  t, float x, float y, float z, float4 dPdx, float4 dPdy) 
# 726
{int volatile ___ = 1;(void)t;(void)x;(void)y;(void)z;(void)dPdx;(void)dPdy;
# 733
::exit(___);}
#if 0
# 726
{ 
# 733
} 
#endif
# 64 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/texture_indirect_functions.h"
template< class T> struct __nv_itex_trait { }; 
# 65
template<> struct __nv_itex_trait< char>  { typedef void type; }; 
# 66
template<> struct __nv_itex_trait< signed char>  { typedef void type; }; 
# 67
template<> struct __nv_itex_trait< char1>  { typedef void type; }; 
# 68
template<> struct __nv_itex_trait< char2>  { typedef void type; }; 
# 69
template<> struct __nv_itex_trait< char4>  { typedef void type; }; 
# 70
template<> struct __nv_itex_trait< unsigned char>  { typedef void type; }; 
# 71
template<> struct __nv_itex_trait< uchar1>  { typedef void type; }; 
# 72
template<> struct __nv_itex_trait< uchar2>  { typedef void type; }; 
# 73
template<> struct __nv_itex_trait< uchar4>  { typedef void type; }; 
# 74
template<> struct __nv_itex_trait< short>  { typedef void type; }; 
# 75
template<> struct __nv_itex_trait< short1>  { typedef void type; }; 
# 76
template<> struct __nv_itex_trait< short2>  { typedef void type; }; 
# 77
template<> struct __nv_itex_trait< short4>  { typedef void type; }; 
# 78
template<> struct __nv_itex_trait< unsigned short>  { typedef void type; }; 
# 79
template<> struct __nv_itex_trait< ushort1>  { typedef void type; }; 
# 80
template<> struct __nv_itex_trait< ushort2>  { typedef void type; }; 
# 81
template<> struct __nv_itex_trait< ushort4>  { typedef void type; }; 
# 82
template<> struct __nv_itex_trait< int>  { typedef void type; }; 
# 83
template<> struct __nv_itex_trait< int1>  { typedef void type; }; 
# 84
template<> struct __nv_itex_trait< int2>  { typedef void type; }; 
# 85
template<> struct __nv_itex_trait< int4>  { typedef void type; }; 
# 86
template<> struct __nv_itex_trait< unsigned>  { typedef void type; }; 
# 87
template<> struct __nv_itex_trait< uint1>  { typedef void type; }; 
# 88
template<> struct __nv_itex_trait< uint2>  { typedef void type; }; 
# 89
template<> struct __nv_itex_trait< uint4>  { typedef void type; }; 
# 100 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/texture_indirect_functions.h"
template<> struct __nv_itex_trait< float>  { typedef void type; }; 
# 101
template<> struct __nv_itex_trait< float1>  { typedef void type; }; 
# 102
template<> struct __nv_itex_trait< float2>  { typedef void type; }; 
# 103
template<> struct __nv_itex_trait< float4>  { typedef void type; }; 
# 107
template< class T> __attribute__((unused)) static typename __nv_itex_trait< T> ::type 
# 108
tex1Dfetch(T *ptr, cudaTextureObject_t obj, int x) 
# 109
{int volatile ___ = 1;(void)ptr;(void)obj;(void)x;
# 113
::exit(___);}
#if 0
# 109
{ 
# 113
} 
#endif
# 115 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/texture_indirect_functions.h"
template< class T> __attribute__((unused)) static T 
# 116
tex1Dfetch(cudaTextureObject_t texObject, int x) 
# 117
{int volatile ___ = 1;(void)texObject;(void)x;
# 123
::exit(___);}
#if 0
# 117
{ 
# 123
} 
#endif
# 125 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/texture_indirect_functions.h"
template< class T> __attribute__((unused)) static typename __nv_itex_trait< T> ::type 
# 126
tex1D(T *ptr, cudaTextureObject_t obj, float x) 
# 127
{int volatile ___ = 1;(void)ptr;(void)obj;(void)x;
# 131
::exit(___);}
#if 0
# 127
{ 
# 131
} 
#endif
# 134 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/texture_indirect_functions.h"
template< class T> __attribute__((unused)) static T 
# 135
tex1D(cudaTextureObject_t texObject, float x) 
# 136
{int volatile ___ = 1;(void)texObject;(void)x;
# 142
::exit(___);}
#if 0
# 136
{ 
# 142
} 
#endif
# 145 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/texture_indirect_functions.h"
template< class T> __attribute__((unused)) static typename __nv_itex_trait< T> ::type 
# 146
tex2D(T *ptr, cudaTextureObject_t obj, float x, float y) 
# 147
{int volatile ___ = 1;(void)ptr;(void)obj;(void)x;(void)y;
# 151
::exit(___);}
#if 0
# 147
{ 
# 151
} 
#endif
# 153 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/texture_indirect_functions.h"
template< class T> __attribute__((unused)) static T 
# 154
tex2D(cudaTextureObject_t texObject, float x, float y) 
# 155
{int volatile ___ = 1;(void)texObject;(void)x;(void)y;
# 161
::exit(___);}
#if 0
# 155
{ 
# 161
} 
#endif
# 164 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/texture_indirect_functions.h"
template< class T> __attribute__((unused)) static typename __nv_itex_trait< T> ::type 
# 165
tex2D(T *ptr, cudaTextureObject_t obj, float x, float y, bool *
# 166
isResident) 
# 167
{int volatile ___ = 1;(void)ptr;(void)obj;(void)x;(void)y;(void)isResident;
# 173
::exit(___);}
#if 0
# 167
{ 
# 173
} 
#endif
# 175 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/texture_indirect_functions.h"
template< class T> __attribute__((unused)) static T 
# 176
tex2D(cudaTextureObject_t texObject, float x, float y, bool *isResident) 
# 177
{int volatile ___ = 1;(void)texObject;(void)x;(void)y;(void)isResident;
# 183
::exit(___);}
#if 0
# 177
{ 
# 183
} 
#endif
# 188 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/texture_indirect_functions.h"
template< class T> __attribute__((unused)) static typename __nv_itex_trait< T> ::type 
# 189
tex3D(T *ptr, cudaTextureObject_t obj, float x, float y, float z) 
# 190
{int volatile ___ = 1;(void)ptr;(void)obj;(void)x;(void)y;(void)z;
# 194
::exit(___);}
#if 0
# 190
{ 
# 194
} 
#endif
# 196 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/texture_indirect_functions.h"
template< class T> __attribute__((unused)) static T 
# 197
tex3D(cudaTextureObject_t texObject, float x, float y, float z) 
# 198
{int volatile ___ = 1;(void)texObject;(void)x;(void)y;(void)z;
# 204
::exit(___);}
#if 0
# 198
{ 
# 204
} 
#endif
# 207 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/texture_indirect_functions.h"
template< class T> __attribute__((unused)) static typename __nv_itex_trait< T> ::type 
# 208
tex3D(T *ptr, cudaTextureObject_t obj, float x, float y, float z, bool *
# 209
isResident) 
# 210
{int volatile ___ = 1;(void)ptr;(void)obj;(void)x;(void)y;(void)z;(void)isResident;
# 216
::exit(___);}
#if 0
# 210
{ 
# 216
} 
#endif
# 218 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/texture_indirect_functions.h"
template< class T> __attribute__((unused)) static T 
# 219
tex3D(cudaTextureObject_t texObject, float x, float y, float z, bool *isResident) 
# 220
{int volatile ___ = 1;(void)texObject;(void)x;(void)y;(void)z;(void)isResident;
# 226
::exit(___);}
#if 0
# 220
{ 
# 226
} 
#endif
# 230 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/texture_indirect_functions.h"
template< class T> __attribute__((unused)) static typename __nv_itex_trait< T> ::type 
# 231
tex1DLayered(T *ptr, cudaTextureObject_t obj, float x, int layer) 
# 232
{int volatile ___ = 1;(void)ptr;(void)obj;(void)x;(void)layer;
# 236
::exit(___);}
#if 0
# 232
{ 
# 236
} 
#endif
# 238 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/texture_indirect_functions.h"
template< class T> __attribute__((unused)) static T 
# 239
tex1DLayered(cudaTextureObject_t texObject, float x, int layer) 
# 240
{int volatile ___ = 1;(void)texObject;(void)x;(void)layer;
# 246
::exit(___);}
#if 0
# 240
{ 
# 246
} 
#endif
# 248 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/texture_indirect_functions.h"
template< class T> __attribute__((unused)) static typename __nv_itex_trait< T> ::type 
# 249
tex2DLayered(T *ptr, cudaTextureObject_t obj, float x, float y, int layer) 
# 250
{int volatile ___ = 1;(void)ptr;(void)obj;(void)x;(void)y;(void)layer;
# 254
::exit(___);}
#if 0
# 250
{ 
# 254
} 
#endif
# 256 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/texture_indirect_functions.h"
template< class T> __attribute__((unused)) static T 
# 257
tex2DLayered(cudaTextureObject_t texObject, float x, float y, int layer) 
# 258
{int volatile ___ = 1;(void)texObject;(void)x;(void)y;(void)layer;
# 264
::exit(___);}
#if 0
# 258
{ 
# 264
} 
#endif
# 267 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/texture_indirect_functions.h"
template< class T> __attribute__((unused)) static typename __nv_itex_trait< T> ::type 
# 268
tex2DLayered(T *ptr, cudaTextureObject_t obj, float x, float y, int layer, bool *isResident) 
# 269
{int volatile ___ = 1;(void)ptr;(void)obj;(void)x;(void)y;(void)layer;(void)isResident;
# 275
::exit(___);}
#if 0
# 269
{ 
# 275
} 
#endif
# 277 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/texture_indirect_functions.h"
template< class T> __attribute__((unused)) static T 
# 278
tex2DLayered(cudaTextureObject_t texObject, float x, float y, int layer, bool *isResident) 
# 279
{int volatile ___ = 1;(void)texObject;(void)x;(void)y;(void)layer;(void)isResident;
# 285
::exit(___);}
#if 0
# 279
{ 
# 285
} 
#endif
# 289 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/texture_indirect_functions.h"
template< class T> __attribute__((unused)) static typename __nv_itex_trait< T> ::type 
# 290
texCubemap(T *ptr, cudaTextureObject_t obj, float x, float y, float z) 
# 291
{int volatile ___ = 1;(void)ptr;(void)obj;(void)x;(void)y;(void)z;
# 295
::exit(___);}
#if 0
# 291
{ 
# 295
} 
#endif
# 298 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/texture_indirect_functions.h"
template< class T> __attribute__((unused)) static T 
# 299
texCubemap(cudaTextureObject_t texObject, float x, float y, float z) 
# 300
{int volatile ___ = 1;(void)texObject;(void)x;(void)y;(void)z;
# 306
::exit(___);}
#if 0
# 300
{ 
# 306
} 
#endif
# 309 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/texture_indirect_functions.h"
template< class T> __attribute__((unused)) static typename __nv_itex_trait< T> ::type 
# 310
texCubemapLayered(T *ptr, cudaTextureObject_t obj, float x, float y, float z, int layer) 
# 311
{int volatile ___ = 1;(void)ptr;(void)obj;(void)x;(void)y;(void)z;(void)layer;
# 315
::exit(___);}
#if 0
# 311
{ 
# 315
} 
#endif
# 317 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/texture_indirect_functions.h"
template< class T> __attribute__((unused)) static T 
# 318
texCubemapLayered(cudaTextureObject_t texObject, float x, float y, float z, int layer) 
# 319
{int volatile ___ = 1;(void)texObject;(void)x;(void)y;(void)z;(void)layer;
# 325
::exit(___);}
#if 0
# 319
{ 
# 325
} 
#endif
# 327 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/texture_indirect_functions.h"
template< class T> __attribute__((unused)) static typename __nv_itex_trait< T> ::type 
# 328
tex2Dgather(T *ptr, cudaTextureObject_t obj, float x, float y, int comp = 0) 
# 329
{int volatile ___ = 1;(void)ptr;(void)obj;(void)x;(void)y;(void)comp;
# 333
::exit(___);}
#if 0
# 329
{ 
# 333
} 
#endif
# 335 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/texture_indirect_functions.h"
template< class T> __attribute__((unused)) static T 
# 336
tex2Dgather(cudaTextureObject_t to, float x, float y, int comp = 0) 
# 337
{int volatile ___ = 1;(void)to;(void)x;(void)y;(void)comp;
# 343
::exit(___);}
#if 0
# 337
{ 
# 343
} 
#endif
# 346 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/texture_indirect_functions.h"
template< class T> __attribute__((unused)) static typename __nv_itex_trait< T> ::type 
# 347
tex2Dgather(T *ptr, cudaTextureObject_t obj, float x, float y, bool *isResident, int comp = 0) 
# 348
{int volatile ___ = 1;(void)ptr;(void)obj;(void)x;(void)y;(void)isResident;(void)comp;
# 354
::exit(___);}
#if 0
# 348
{ 
# 354
} 
#endif
# 356 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/texture_indirect_functions.h"
template< class T> __attribute__((unused)) static T 
# 357
tex2Dgather(cudaTextureObject_t to, float x, float y, bool *isResident, int comp = 0) 
# 358
{int volatile ___ = 1;(void)to;(void)x;(void)y;(void)isResident;(void)comp;
# 364
::exit(___);}
#if 0
# 358
{ 
# 364
} 
#endif
# 368 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/texture_indirect_functions.h"
template< class T> __attribute__((unused)) static typename __nv_itex_trait< T> ::type 
# 369
tex1DLod(T *ptr, cudaTextureObject_t obj, float x, float level) 
# 370
{int volatile ___ = 1;(void)ptr;(void)obj;(void)x;(void)level;
# 374
::exit(___);}
#if 0
# 370
{ 
# 374
} 
#endif
# 376 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/texture_indirect_functions.h"
template< class T> __attribute__((unused)) static T 
# 377
tex1DLod(cudaTextureObject_t texObject, float x, float level) 
# 378
{int volatile ___ = 1;(void)texObject;(void)x;(void)level;
# 384
::exit(___);}
#if 0
# 378
{ 
# 384
} 
#endif
# 387 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/texture_indirect_functions.h"
template< class T> __attribute__((unused)) static typename __nv_itex_trait< T> ::type 
# 388
tex2DLod(T *ptr, cudaTextureObject_t obj, float x, float y, float level) 
# 389
{int volatile ___ = 1;(void)ptr;(void)obj;(void)x;(void)y;(void)level;
# 393
::exit(___);}
#if 0
# 389
{ 
# 393
} 
#endif
# 395 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/texture_indirect_functions.h"
template< class T> __attribute__((unused)) static T 
# 396
tex2DLod(cudaTextureObject_t texObject, float x, float y, float level) 
# 397
{int volatile ___ = 1;(void)texObject;(void)x;(void)y;(void)level;
# 403
::exit(___);}
#if 0
# 397
{ 
# 403
} 
#endif
# 407 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/texture_indirect_functions.h"
template< class T> __attribute__((unused)) static typename __nv_itex_trait< T> ::type 
# 408
tex2DLod(T *ptr, cudaTextureObject_t obj, float x, float y, float level, bool *isResident) 
# 409
{int volatile ___ = 1;(void)ptr;(void)obj;(void)x;(void)y;(void)level;(void)isResident;
# 415
::exit(___);}
#if 0
# 409
{ 
# 415
} 
#endif
# 417 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/texture_indirect_functions.h"
template< class T> __attribute__((unused)) static T 
# 418
tex2DLod(cudaTextureObject_t texObject, float x, float y, float level, bool *isResident) 
# 419
{int volatile ___ = 1;(void)texObject;(void)x;(void)y;(void)level;(void)isResident;
# 425
::exit(___);}
#if 0
# 419
{ 
# 425
} 
#endif
# 430 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/texture_indirect_functions.h"
template< class T> __attribute__((unused)) static typename __nv_itex_trait< T> ::type 
# 431
tex3DLod(T *ptr, cudaTextureObject_t obj, float x, float y, float z, float level) 
# 432
{int volatile ___ = 1;(void)ptr;(void)obj;(void)x;(void)y;(void)z;(void)level;
# 436
::exit(___);}
#if 0
# 432
{ 
# 436
} 
#endif
# 438 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/texture_indirect_functions.h"
template< class T> __attribute__((unused)) static T 
# 439
tex3DLod(cudaTextureObject_t texObject, float x, float y, float z, float level) 
# 440
{int volatile ___ = 1;(void)texObject;(void)x;(void)y;(void)z;(void)level;
# 446
::exit(___);}
#if 0
# 440
{ 
# 446
} 
#endif
# 449 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/texture_indirect_functions.h"
template< class T> __attribute__((unused)) static typename __nv_itex_trait< T> ::type 
# 450
tex3DLod(T *ptr, cudaTextureObject_t obj, float x, float y, float z, float level, bool *isResident) 
# 451
{int volatile ___ = 1;(void)ptr;(void)obj;(void)x;(void)y;(void)z;(void)level;(void)isResident;
# 457
::exit(___);}
#if 0
# 451
{ 
# 457
} 
#endif
# 459 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/texture_indirect_functions.h"
template< class T> __attribute__((unused)) static T 
# 460
tex3DLod(cudaTextureObject_t texObject, float x, float y, float z, float level, bool *isResident) 
# 461
{int volatile ___ = 1;(void)texObject;(void)x;(void)y;(void)z;(void)level;(void)isResident;
# 467
::exit(___);}
#if 0
# 461
{ 
# 467
} 
#endif
# 472 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/texture_indirect_functions.h"
template< class T> __attribute__((unused)) static typename __nv_itex_trait< T> ::type 
# 473
tex1DLayeredLod(T *ptr, cudaTextureObject_t obj, float x, int layer, float level) 
# 474
{int volatile ___ = 1;(void)ptr;(void)obj;(void)x;(void)layer;(void)level;
# 478
::exit(___);}
#if 0
# 474
{ 
# 478
} 
#endif
# 480 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/texture_indirect_functions.h"
template< class T> __attribute__((unused)) static T 
# 481
tex1DLayeredLod(cudaTextureObject_t texObject, float x, int layer, float level) 
# 482
{int volatile ___ = 1;(void)texObject;(void)x;(void)layer;(void)level;
# 488
::exit(___);}
#if 0
# 482
{ 
# 488
} 
#endif
# 491 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/texture_indirect_functions.h"
template< class T> __attribute__((unused)) static typename __nv_itex_trait< T> ::type 
# 492
tex2DLayeredLod(T *ptr, cudaTextureObject_t obj, float x, float y, int layer, float level) 
# 493
{int volatile ___ = 1;(void)ptr;(void)obj;(void)x;(void)y;(void)layer;(void)level;
# 497
::exit(___);}
#if 0
# 493
{ 
# 497
} 
#endif
# 499 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/texture_indirect_functions.h"
template< class T> __attribute__((unused)) static T 
# 500
tex2DLayeredLod(cudaTextureObject_t texObject, float x, float y, int layer, float level) 
# 501
{int volatile ___ = 1;(void)texObject;(void)x;(void)y;(void)layer;(void)level;
# 507
::exit(___);}
#if 0
# 501
{ 
# 507
} 
#endif
# 510 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/texture_indirect_functions.h"
template< class T> __attribute__((unused)) static typename __nv_itex_trait< T> ::type 
# 511
tex2DLayeredLod(T *ptr, cudaTextureObject_t obj, float x, float y, int layer, float level, bool *isResident) 
# 512
{int volatile ___ = 1;(void)ptr;(void)obj;(void)x;(void)y;(void)layer;(void)level;(void)isResident;
# 518
::exit(___);}
#if 0
# 512
{ 
# 518
} 
#endif
# 520 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/texture_indirect_functions.h"
template< class T> __attribute__((unused)) static T 
# 521
tex2DLayeredLod(cudaTextureObject_t texObject, float x, float y, int layer, float level, bool *isResident) 
# 522
{int volatile ___ = 1;(void)texObject;(void)x;(void)y;(void)layer;(void)level;(void)isResident;
# 528
::exit(___);}
#if 0
# 522
{ 
# 528
} 
#endif
# 531 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/texture_indirect_functions.h"
template< class T> __attribute__((unused)) static typename __nv_itex_trait< T> ::type 
# 532
texCubemapLod(T *ptr, cudaTextureObject_t obj, float x, float y, float z, float level) 
# 533
{int volatile ___ = 1;(void)ptr;(void)obj;(void)x;(void)y;(void)z;(void)level;
# 537
::exit(___);}
#if 0
# 533
{ 
# 537
} 
#endif
# 539 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/texture_indirect_functions.h"
template< class T> __attribute__((unused)) static T 
# 540
texCubemapLod(cudaTextureObject_t texObject, float x, float y, float z, float level) 
# 541
{int volatile ___ = 1;(void)texObject;(void)x;(void)y;(void)z;(void)level;
# 547
::exit(___);}
#if 0
# 541
{ 
# 547
} 
#endif
# 550 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/texture_indirect_functions.h"
template< class T> __attribute__((unused)) static typename __nv_itex_trait< T> ::type 
# 551
texCubemapGrad(T *ptr, cudaTextureObject_t obj, float x, float y, float z, float4 dPdx, float4 dPdy) 
# 552
{int volatile ___ = 1;(void)ptr;(void)obj;(void)x;(void)y;(void)z;(void)dPdx;(void)dPdy;
# 556
::exit(___);}
#if 0
# 552
{ 
# 556
} 
#endif
# 558 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/texture_indirect_functions.h"
template< class T> __attribute__((unused)) static T 
# 559
texCubemapGrad(cudaTextureObject_t texObject, float x, float y, float z, float4 dPdx, float4 dPdy) 
# 560
{int volatile ___ = 1;(void)texObject;(void)x;(void)y;(void)z;(void)dPdx;(void)dPdy;
# 566
::exit(___);}
#if 0
# 560
{ 
# 566
} 
#endif
# 568 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/texture_indirect_functions.h"
template< class T> __attribute__((unused)) static typename __nv_itex_trait< T> ::type 
# 569
texCubemapLayeredLod(T *ptr, cudaTextureObject_t obj, float x, float y, float z, int layer, float level) 
# 570
{int volatile ___ = 1;(void)ptr;(void)obj;(void)x;(void)y;(void)z;(void)layer;(void)level;
# 574
::exit(___);}
#if 0
# 570
{ 
# 574
} 
#endif
# 576 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/texture_indirect_functions.h"
template< class T> __attribute__((unused)) static T 
# 577
texCubemapLayeredLod(cudaTextureObject_t texObject, float x, float y, float z, int layer, float level) 
# 578
{int volatile ___ = 1;(void)texObject;(void)x;(void)y;(void)z;(void)layer;(void)level;
# 584
::exit(___);}
#if 0
# 578
{ 
# 584
} 
#endif
# 586 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/texture_indirect_functions.h"
template< class T> __attribute__((unused)) static typename __nv_itex_trait< T> ::type 
# 587
tex1DGrad(T *ptr, cudaTextureObject_t obj, float x, float dPdx, float dPdy) 
# 588
{int volatile ___ = 1;(void)ptr;(void)obj;(void)x;(void)dPdx;(void)dPdy;
# 592
::exit(___);}
#if 0
# 588
{ 
# 592
} 
#endif
# 594 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/texture_indirect_functions.h"
template< class T> __attribute__((unused)) static T 
# 595
tex1DGrad(cudaTextureObject_t texObject, float x, float dPdx, float dPdy) 
# 596
{int volatile ___ = 1;(void)texObject;(void)x;(void)dPdx;(void)dPdy;
# 602
::exit(___);}
#if 0
# 596
{ 
# 602
} 
#endif
# 605 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/texture_indirect_functions.h"
template< class T> __attribute__((unused)) static typename __nv_itex_trait< T> ::type 
# 606
tex2DGrad(T *ptr, cudaTextureObject_t obj, float x, float y, float2 dPdx, float2 dPdy) 
# 607
{int volatile ___ = 1;(void)ptr;(void)obj;(void)x;(void)y;(void)dPdx;(void)dPdy;
# 612
::exit(___);}
#if 0
# 607
{ 
# 612
} 
#endif
# 614 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/texture_indirect_functions.h"
template< class T> __attribute__((unused)) static T 
# 615
tex2DGrad(cudaTextureObject_t texObject, float x, float y, float2 dPdx, float2 dPdy) 
# 616
{int volatile ___ = 1;(void)texObject;(void)x;(void)y;(void)dPdx;(void)dPdy;
# 622
::exit(___);}
#if 0
# 616
{ 
# 622
} 
#endif
# 625 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/texture_indirect_functions.h"
template< class T> __attribute__((unused)) static typename __nv_itex_trait< T> ::type 
# 626
tex2DGrad(T *ptr, cudaTextureObject_t obj, float x, float y, float2 dPdx, float2 dPdy, bool *isResident) 
# 627
{int volatile ___ = 1;(void)ptr;(void)obj;(void)x;(void)y;(void)dPdx;(void)dPdy;(void)isResident;
# 634
::exit(___);}
#if 0
# 627
{ 
# 634
} 
#endif
# 636 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/texture_indirect_functions.h"
template< class T> __attribute__((unused)) static T 
# 637
tex2DGrad(cudaTextureObject_t texObject, float x, float y, float2 dPdx, float2 dPdy, bool *isResident) 
# 638
{int volatile ___ = 1;(void)texObject;(void)x;(void)y;(void)dPdx;(void)dPdy;(void)isResident;
# 644
::exit(___);}
#if 0
# 638
{ 
# 644
} 
#endif
# 648 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/texture_indirect_functions.h"
template< class T> __attribute__((unused)) static typename __nv_itex_trait< T> ::type 
# 649
tex3DGrad(T *ptr, cudaTextureObject_t obj, float x, float y, float z, float4 dPdx, float4 dPdy) 
# 650
{int volatile ___ = 1;(void)ptr;(void)obj;(void)x;(void)y;(void)z;(void)dPdx;(void)dPdy;
# 654
::exit(___);}
#if 0
# 650
{ 
# 654
} 
#endif
# 656 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/texture_indirect_functions.h"
template< class T> __attribute__((unused)) static T 
# 657
tex3DGrad(cudaTextureObject_t texObject, float x, float y, float z, float4 dPdx, float4 dPdy) 
# 658
{int volatile ___ = 1;(void)texObject;(void)x;(void)y;(void)z;(void)dPdx;(void)dPdy;
# 664
::exit(___);}
#if 0
# 658
{ 
# 664
} 
#endif
# 667 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/texture_indirect_functions.h"
template< class T> __attribute__((unused)) static typename __nv_itex_trait< T> ::type 
# 668
tex3DGrad(T *ptr, cudaTextureObject_t obj, float x, float y, float z, float4 dPdx, float4 dPdy, bool *isResident) 
# 669
{int volatile ___ = 1;(void)ptr;(void)obj;(void)x;(void)y;(void)z;(void)dPdx;(void)dPdy;(void)isResident;
# 675
::exit(___);}
#if 0
# 669
{ 
# 675
} 
#endif
# 677 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/texture_indirect_functions.h"
template< class T> __attribute__((unused)) static T 
# 678
tex3DGrad(cudaTextureObject_t texObject, float x, float y, float z, float4 dPdx, float4 dPdy, bool *isResident) 
# 679
{int volatile ___ = 1;(void)texObject;(void)x;(void)y;(void)z;(void)dPdx;(void)dPdy;(void)isResident;
# 685
::exit(___);}
#if 0
# 679
{ 
# 685
} 
#endif
# 690 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/texture_indirect_functions.h"
template< class T> __attribute__((unused)) static typename __nv_itex_trait< T> ::type 
# 691
tex1DLayeredGrad(T *ptr, cudaTextureObject_t obj, float x, int layer, float dPdx, float dPdy) 
# 692
{int volatile ___ = 1;(void)ptr;(void)obj;(void)x;(void)layer;(void)dPdx;(void)dPdy;
# 696
::exit(___);}
#if 0
# 692
{ 
# 696
} 
#endif
# 698 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/texture_indirect_functions.h"
template< class T> __attribute__((unused)) static T 
# 699
tex1DLayeredGrad(cudaTextureObject_t texObject, float x, int layer, float dPdx, float dPdy) 
# 700
{int volatile ___ = 1;(void)texObject;(void)x;(void)layer;(void)dPdx;(void)dPdy;
# 706
::exit(___);}
#if 0
# 700
{ 
# 706
} 
#endif
# 709 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/texture_indirect_functions.h"
template< class T> __attribute__((unused)) static typename __nv_itex_trait< T> ::type 
# 710
tex2DLayeredGrad(T *ptr, cudaTextureObject_t obj, float x, float y, int layer, float2 dPdx, float2 dPdy) 
# 711
{int volatile ___ = 1;(void)ptr;(void)obj;(void)x;(void)y;(void)layer;(void)dPdx;(void)dPdy;
# 715
::exit(___);}
#if 0
# 711
{ 
# 715
} 
#endif
# 717 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/texture_indirect_functions.h"
template< class T> __attribute__((unused)) static T 
# 718
tex2DLayeredGrad(cudaTextureObject_t texObject, float x, float y, int layer, float2 dPdx, float2 dPdy) 
# 719
{int volatile ___ = 1;(void)texObject;(void)x;(void)y;(void)layer;(void)dPdx;(void)dPdy;
# 725
::exit(___);}
#if 0
# 719
{ 
# 725
} 
#endif
# 728 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/texture_indirect_functions.h"
template< class T> __attribute__((unused)) static typename __nv_itex_trait< T> ::type 
# 729
tex2DLayeredGrad(T *ptr, cudaTextureObject_t obj, float x, float y, int layer, float2 dPdx, float2 dPdy, bool *isResident) 
# 730
{int volatile ___ = 1;(void)ptr;(void)obj;(void)x;(void)y;(void)layer;(void)dPdx;(void)dPdy;(void)isResident;
# 736
::exit(___);}
#if 0
# 730
{ 
# 736
} 
#endif
# 738 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/texture_indirect_functions.h"
template< class T> __attribute__((unused)) static T 
# 739
tex2DLayeredGrad(cudaTextureObject_t texObject, float x, float y, int layer, float2 dPdx, float2 dPdy, bool *isResident) 
# 740
{int volatile ___ = 1;(void)texObject;(void)x;(void)y;(void)layer;(void)dPdx;(void)dPdy;(void)isResident;
# 746
::exit(___);}
#if 0
# 740
{ 
# 746
} 
#endif
# 750 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/texture_indirect_functions.h"
template< class T> __attribute__((unused)) static typename __nv_itex_trait< T> ::type 
# 751
texCubemapLayeredGrad(T *ptr, cudaTextureObject_t obj, float x, float y, float z, int layer, float4 dPdx, float4 dPdy) 
# 752
{int volatile ___ = 1;(void)ptr;(void)obj;(void)x;(void)y;(void)z;(void)layer;(void)dPdx;(void)dPdy;
# 756
::exit(___);}
#if 0
# 752
{ 
# 756
} 
#endif
# 758 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/texture_indirect_functions.h"
template< class T> __attribute__((unused)) static T 
# 759
texCubemapLayeredGrad(cudaTextureObject_t texObject, float x, float y, float z, int layer, float4 dPdx, float4 dPdy) 
# 760
{int volatile ___ = 1;(void)texObject;(void)x;(void)y;(void)z;(void)layer;(void)dPdx;(void)dPdy;
# 766
::exit(___);}
#if 0
# 760
{ 
# 766
} 
#endif
# 59 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/surface_indirect_functions.h"
template< class T> struct __nv_isurf_trait { }; 
# 60
template<> struct __nv_isurf_trait< char>  { typedef void type; }; 
# 61
template<> struct __nv_isurf_trait< signed char>  { typedef void type; }; 
# 62
template<> struct __nv_isurf_trait< char1>  { typedef void type; }; 
# 63
template<> struct __nv_isurf_trait< unsigned char>  { typedef void type; }; 
# 64
template<> struct __nv_isurf_trait< uchar1>  { typedef void type; }; 
# 65
template<> struct __nv_isurf_trait< short>  { typedef void type; }; 
# 66
template<> struct __nv_isurf_trait< short1>  { typedef void type; }; 
# 67
template<> struct __nv_isurf_trait< unsigned short>  { typedef void type; }; 
# 68
template<> struct __nv_isurf_trait< ushort1>  { typedef void type; }; 
# 69
template<> struct __nv_isurf_trait< int>  { typedef void type; }; 
# 70
template<> struct __nv_isurf_trait< int1>  { typedef void type; }; 
# 71
template<> struct __nv_isurf_trait< unsigned>  { typedef void type; }; 
# 72
template<> struct __nv_isurf_trait< uint1>  { typedef void type; }; 
# 73
template<> struct __nv_isurf_trait< long long>  { typedef void type; }; 
# 74
template<> struct __nv_isurf_trait< longlong1>  { typedef void type; }; 
# 75
template<> struct __nv_isurf_trait< unsigned long long>  { typedef void type; }; 
# 76
template<> struct __nv_isurf_trait< ulonglong1>  { typedef void type; }; 
# 77
template<> struct __nv_isurf_trait< float>  { typedef void type; }; 
# 78
template<> struct __nv_isurf_trait< float1>  { typedef void type; }; 
# 80
template<> struct __nv_isurf_trait< char2>  { typedef void type; }; 
# 81
template<> struct __nv_isurf_trait< uchar2>  { typedef void type; }; 
# 82
template<> struct __nv_isurf_trait< short2>  { typedef void type; }; 
# 83
template<> struct __nv_isurf_trait< ushort2>  { typedef void type; }; 
# 84
template<> struct __nv_isurf_trait< int2>  { typedef void type; }; 
# 85
template<> struct __nv_isurf_trait< uint2>  { typedef void type; }; 
# 86
template<> struct __nv_isurf_trait< longlong2>  { typedef void type; }; 
# 87
template<> struct __nv_isurf_trait< ulonglong2>  { typedef void type; }; 
# 88
template<> struct __nv_isurf_trait< float2>  { typedef void type; }; 
# 90
template<> struct __nv_isurf_trait< char4>  { typedef void type; }; 
# 91
template<> struct __nv_isurf_trait< uchar4>  { typedef void type; }; 
# 92
template<> struct __nv_isurf_trait< short4>  { typedef void type; }; 
# 93
template<> struct __nv_isurf_trait< ushort4>  { typedef void type; }; 
# 94
template<> struct __nv_isurf_trait< int4>  { typedef void type; }; 
# 95
template<> struct __nv_isurf_trait< uint4>  { typedef void type; }; 
# 96
template<> struct __nv_isurf_trait< float4>  { typedef void type; }; 
# 99
template< class T> __attribute__((unused)) static typename __nv_isurf_trait< T> ::type 
# 100
surf1Dread(T *ptr, cudaSurfaceObject_t obj, int x, cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap) 
# 101
{int volatile ___ = 1;(void)ptr;(void)obj;(void)x;(void)mode;
# 105
::exit(___);}
#if 0
# 101
{ 
# 105
} 
#endif
# 107 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/surface_indirect_functions.h"
template< class T> __attribute__((unused)) static T 
# 108
surf1Dread(cudaSurfaceObject_t surfObject, int x, cudaSurfaceBoundaryMode boundaryMode = cudaBoundaryModeTrap) 
# 109
{int volatile ___ = 1;(void)surfObject;(void)x;(void)boundaryMode;
# 115
::exit(___);}
#if 0
# 109
{ 
# 115
} 
#endif
# 117 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/surface_indirect_functions.h"
template< class T> __attribute__((unused)) static typename __nv_isurf_trait< T> ::type 
# 118
surf2Dread(T *ptr, cudaSurfaceObject_t obj, int x, int y, cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap) 
# 119
{int volatile ___ = 1;(void)ptr;(void)obj;(void)x;(void)y;(void)mode;
# 123
::exit(___);}
#if 0
# 119
{ 
# 123
} 
#endif
# 125 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/surface_indirect_functions.h"
template< class T> __attribute__((unused)) static T 
# 126
surf2Dread(cudaSurfaceObject_t surfObject, int x, int y, cudaSurfaceBoundaryMode boundaryMode = cudaBoundaryModeTrap) 
# 127
{int volatile ___ = 1;(void)surfObject;(void)x;(void)y;(void)boundaryMode;
# 133
::exit(___);}
#if 0
# 127
{ 
# 133
} 
#endif
# 136 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/surface_indirect_functions.h"
template< class T> __attribute__((unused)) static typename __nv_isurf_trait< T> ::type 
# 137
surf3Dread(T *ptr, cudaSurfaceObject_t obj, int x, int y, int z, cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap) 
# 138
{int volatile ___ = 1;(void)ptr;(void)obj;(void)x;(void)y;(void)z;(void)mode;
# 142
::exit(___);}
#if 0
# 138
{ 
# 142
} 
#endif
# 144 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/surface_indirect_functions.h"
template< class T> __attribute__((unused)) static T 
# 145
surf3Dread(cudaSurfaceObject_t surfObject, int x, int y, int z, cudaSurfaceBoundaryMode boundaryMode = cudaBoundaryModeTrap) 
# 146
{int volatile ___ = 1;(void)surfObject;(void)x;(void)y;(void)z;(void)boundaryMode;
# 152
::exit(___);}
#if 0
# 146
{ 
# 152
} 
#endif
# 154 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/surface_indirect_functions.h"
template< class T> __attribute__((unused)) static typename __nv_isurf_trait< T> ::type 
# 155
surf1DLayeredread(T *ptr, cudaSurfaceObject_t obj, int x, int layer, cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap) 
# 156
{int volatile ___ = 1;(void)ptr;(void)obj;(void)x;(void)layer;(void)mode;
# 160
::exit(___);}
#if 0
# 156
{ 
# 160
} 
#endif
# 162 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/surface_indirect_functions.h"
template< class T> __attribute__((unused)) static T 
# 163
surf1DLayeredread(cudaSurfaceObject_t surfObject, int x, int layer, cudaSurfaceBoundaryMode boundaryMode = cudaBoundaryModeTrap) 
# 164
{int volatile ___ = 1;(void)surfObject;(void)x;(void)layer;(void)boundaryMode;
# 170
::exit(___);}
#if 0
# 164
{ 
# 170
} 
#endif
# 172 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/surface_indirect_functions.h"
template< class T> __attribute__((unused)) static typename __nv_isurf_trait< T> ::type 
# 173
surf2DLayeredread(T *ptr, cudaSurfaceObject_t obj, int x, int y, int layer, cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap) 
# 174
{int volatile ___ = 1;(void)ptr;(void)obj;(void)x;(void)y;(void)layer;(void)mode;
# 178
::exit(___);}
#if 0
# 174
{ 
# 178
} 
#endif
# 180 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/surface_indirect_functions.h"
template< class T> __attribute__((unused)) static T 
# 181
surf2DLayeredread(cudaSurfaceObject_t surfObject, int x, int y, int layer, cudaSurfaceBoundaryMode boundaryMode = cudaBoundaryModeTrap) 
# 182
{int volatile ___ = 1;(void)surfObject;(void)x;(void)y;(void)layer;(void)boundaryMode;
# 188
::exit(___);}
#if 0
# 182
{ 
# 188
} 
#endif
# 190 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/surface_indirect_functions.h"
template< class T> __attribute__((unused)) static typename __nv_isurf_trait< T> ::type 
# 191
surfCubemapread(T *ptr, cudaSurfaceObject_t obj, int x, int y, int face, cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap) 
# 192
{int volatile ___ = 1;(void)ptr;(void)obj;(void)x;(void)y;(void)face;(void)mode;
# 196
::exit(___);}
#if 0
# 192
{ 
# 196
} 
#endif
# 198 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/surface_indirect_functions.h"
template< class T> __attribute__((unused)) static T 
# 199
surfCubemapread(cudaSurfaceObject_t surfObject, int x, int y, int face, cudaSurfaceBoundaryMode boundaryMode = cudaBoundaryModeTrap) 
# 200
{int volatile ___ = 1;(void)surfObject;(void)x;(void)y;(void)face;(void)boundaryMode;
# 206
::exit(___);}
#if 0
# 200
{ 
# 206
} 
#endif
# 208 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/surface_indirect_functions.h"
template< class T> __attribute__((unused)) static typename __nv_isurf_trait< T> ::type 
# 209
surfCubemapLayeredread(T *ptr, cudaSurfaceObject_t obj, int x, int y, int layerface, cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap) 
# 210
{int volatile ___ = 1;(void)ptr;(void)obj;(void)x;(void)y;(void)layerface;(void)mode;
# 214
::exit(___);}
#if 0
# 210
{ 
# 214
} 
#endif
# 216 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/surface_indirect_functions.h"
template< class T> __attribute__((unused)) static T 
# 217
surfCubemapLayeredread(cudaSurfaceObject_t surfObject, int x, int y, int layerface, cudaSurfaceBoundaryMode boundaryMode = cudaBoundaryModeTrap) 
# 218
{int volatile ___ = 1;(void)surfObject;(void)x;(void)y;(void)layerface;(void)boundaryMode;
# 224
::exit(___);}
#if 0
# 218
{ 
# 224
} 
#endif
# 226 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/surface_indirect_functions.h"
template< class T> __attribute__((unused)) static typename __nv_isurf_trait< T> ::type 
# 227
surf1Dwrite(T val, cudaSurfaceObject_t obj, int x, cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap) 
# 228
{int volatile ___ = 1;(void)val;(void)obj;(void)x;(void)mode;
# 232
::exit(___);}
#if 0
# 228
{ 
# 232
} 
#endif
# 234 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/surface_indirect_functions.h"
template< class T> __attribute__((unused)) static typename __nv_isurf_trait< T> ::type 
# 235
surf2Dwrite(T val, cudaSurfaceObject_t obj, int x, int y, cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap) 
# 236
{int volatile ___ = 1;(void)val;(void)obj;(void)x;(void)y;(void)mode;
# 240
::exit(___);}
#if 0
# 236
{ 
# 240
} 
#endif
# 242 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/surface_indirect_functions.h"
template< class T> __attribute__((unused)) static typename __nv_isurf_trait< T> ::type 
# 243
surf3Dwrite(T val, cudaSurfaceObject_t obj, int x, int y, int z, cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap) 
# 244
{int volatile ___ = 1;(void)val;(void)obj;(void)x;(void)y;(void)z;(void)mode;
# 248
::exit(___);}
#if 0
# 244
{ 
# 248
} 
#endif
# 250 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/surface_indirect_functions.h"
template< class T> __attribute__((unused)) static typename __nv_isurf_trait< T> ::type 
# 251
surf1DLayeredwrite(T val, cudaSurfaceObject_t obj, int x, int layer, cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap) 
# 252
{int volatile ___ = 1;(void)val;(void)obj;(void)x;(void)layer;(void)mode;
# 256
::exit(___);}
#if 0
# 252
{ 
# 256
} 
#endif
# 258 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/surface_indirect_functions.h"
template< class T> __attribute__((unused)) static typename __nv_isurf_trait< T> ::type 
# 259
surf2DLayeredwrite(T val, cudaSurfaceObject_t obj, int x, int y, int layer, cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap) 
# 260
{int volatile ___ = 1;(void)val;(void)obj;(void)x;(void)y;(void)layer;(void)mode;
# 264
::exit(___);}
#if 0
# 260
{ 
# 264
} 
#endif
# 266 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/surface_indirect_functions.h"
template< class T> __attribute__((unused)) static typename __nv_isurf_trait< T> ::type 
# 267
surfCubemapwrite(T val, cudaSurfaceObject_t obj, int x, int y, int face, cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap) 
# 268
{int volatile ___ = 1;(void)val;(void)obj;(void)x;(void)y;(void)face;(void)mode;
# 272
::exit(___);}
#if 0
# 268
{ 
# 272
} 
#endif
# 274 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/surface_indirect_functions.h"
template< class T> __attribute__((unused)) static typename __nv_isurf_trait< T> ::type 
# 275
surfCubemapLayeredwrite(T val, cudaSurfaceObject_t obj, int x, int y, int layerface, cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap) 
# 276
{int volatile ___ = 1;(void)val;(void)obj;(void)x;(void)y;(void)layerface;(void)mode;
# 280
::exit(___);}
#if 0
# 276
{ 
# 280
} 
#endif
# 3295 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/crt/device_functions.h"
extern "C" unsigned __cudaPushCallConfiguration(dim3 gridDim, dim3 blockDim, size_t sharedMem = 0, CUstream_st * stream = 0); 
# 68 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/device_launch_parameters.h"
extern "C" {
# 71
extern const uint3 __device_builtin_variable_threadIdx; 
# 72
extern const uint3 __device_builtin_variable_blockIdx; 
# 73
extern const dim3 __device_builtin_variable_blockDim; 
# 74
extern const dim3 __device_builtin_variable_gridDim; 
# 75
extern const int __device_builtin_variable_warpSize; 
# 80
}
# 201 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/cuda_runtime.h"
template< class T> static inline cudaError_t 
# 202
cudaLaunchKernel(const T *
# 203
func, dim3 
# 204
gridDim, dim3 
# 205
blockDim, void **
# 206
args, size_t 
# 207
sharedMem = 0, cudaStream_t 
# 208
stream = 0) 
# 210
{ 
# 211
return ::cudaLaunchKernel((const void *)func, gridDim, blockDim, args, sharedMem, stream); 
# 212
} 
# 263 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/cuda_runtime.h"
template< class T> static inline cudaError_t 
# 264
cudaLaunchCooperativeKernel(const T *
# 265
func, dim3 
# 266
gridDim, dim3 
# 267
blockDim, void **
# 268
args, size_t 
# 269
sharedMem = 0, cudaStream_t 
# 270
stream = 0) 
# 272
{ 
# 273
return ::cudaLaunchCooperativeKernel((const void *)func, gridDim, blockDim, args, sharedMem, stream); 
# 274
} 
# 307 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/cuda_runtime.h"
static inline cudaError_t cudaEventCreate(cudaEvent_t *
# 308
event, unsigned 
# 309
flags) 
# 311
{ 
# 312
return ::cudaEventCreateWithFlags(event, flags); 
# 313
} 
# 372 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/cuda_runtime.h"
static inline cudaError_t cudaMallocHost(void **
# 373
ptr, size_t 
# 374
size, unsigned 
# 375
flags) 
# 377
{ 
# 378
return ::cudaHostAlloc(ptr, size, flags); 
# 379
} 
# 381
template< class T> static inline cudaError_t 
# 382
cudaHostAlloc(T **
# 383
ptr, size_t 
# 384
size, unsigned 
# 385
flags) 
# 387
{ 
# 388
return ::cudaHostAlloc((void **)((void *)ptr), size, flags); 
# 389
} 
# 391
template< class T> static inline cudaError_t 
# 392
cudaHostGetDevicePointer(T **
# 393
pDevice, void *
# 394
pHost, unsigned 
# 395
flags) 
# 397
{ 
# 398
return ::cudaHostGetDevicePointer((void **)((void *)pDevice), pHost, flags); 
# 399
} 
# 501 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/cuda_runtime.h"
template< class T> static inline cudaError_t 
# 502
cudaMallocManaged(T **
# 503
devPtr, size_t 
# 504
size, unsigned 
# 505
flags = 1) 
# 507
{ 
# 508
return ::cudaMallocManaged((void **)((void *)devPtr), size, flags); 
# 509
} 
# 591 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/cuda_runtime.h"
template< class T> static inline cudaError_t 
# 592
cudaStreamAttachMemAsync(cudaStream_t 
# 593
stream, T *
# 594
devPtr, size_t 
# 595
length = 0, unsigned 
# 596
flags = 4) 
# 598
{ 
# 599
return ::cudaStreamAttachMemAsync(stream, (void *)devPtr, length, flags); 
# 600
} 
# 602
template< class T> inline cudaError_t 
# 603
cudaMalloc(T **
# 604
devPtr, size_t 
# 605
size) 
# 607
{ 
# 608
return ::cudaMalloc((void **)((void *)devPtr), size); 
# 609
} 
# 611
template< class T> static inline cudaError_t 
# 612
cudaMallocHost(T **
# 613
ptr, size_t 
# 614
size, unsigned 
# 615
flags = 0) 
# 617
{ 
# 618
return cudaMallocHost((void **)((void *)ptr), size, flags); 
# 619
} 
# 621
template< class T> static inline cudaError_t 
# 622
cudaMallocPitch(T **
# 623
devPtr, size_t *
# 624
pitch, size_t 
# 625
width, size_t 
# 626
height) 
# 628
{ 
# 629
return ::cudaMallocPitch((void **)((void *)devPtr), pitch, width, height); 
# 630
} 
# 641 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/cuda_runtime.h"
static inline cudaError_t cudaMallocAsync(void **
# 642
ptr, size_t 
# 643
size, cudaMemPool_t 
# 644
memPool, cudaStream_t 
# 645
stream) 
# 647
{ 
# 648
return ::cudaMallocFromPoolAsync(ptr, size, memPool, stream); 
# 649
} 
# 651
template< class T> static inline cudaError_t 
# 652
cudaMallocAsync(T **
# 653
ptr, size_t 
# 654
size, cudaMemPool_t 
# 655
memPool, cudaStream_t 
# 656
stream) 
# 658
{ 
# 659
return ::cudaMallocFromPoolAsync((void **)((void *)ptr), size, memPool, stream); 
# 660
} 
# 662
template< class T> static inline cudaError_t 
# 663
cudaMallocAsync(T **
# 664
ptr, size_t 
# 665
size, cudaStream_t 
# 666
stream) 
# 668
{ 
# 669
return ::cudaMallocAsync((void **)((void *)ptr), size, stream); 
# 670
} 
# 672
template< class T> static inline cudaError_t 
# 673
cudaMallocFromPoolAsync(T **
# 674
ptr, size_t 
# 675
size, cudaMemPool_t 
# 676
memPool, cudaStream_t 
# 677
stream) 
# 679
{ 
# 680
return ::cudaMallocFromPoolAsync((void **)((void *)ptr), size, memPool, stream); 
# 681
} 
# 720 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/cuda_runtime.h"
template< class T> static inline cudaError_t 
# 721
cudaMemcpyToSymbol(const T &
# 722
symbol, const void *
# 723
src, size_t 
# 724
count, size_t 
# 725
offset = 0, cudaMemcpyKind 
# 726
kind = cudaMemcpyHostToDevice) 
# 728
{ 
# 729
return ::cudaMemcpyToSymbol((const void *)(&symbol), src, count, offset, kind); 
# 730
} 
# 774 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/cuda_runtime.h"
template< class T> static inline cudaError_t 
# 775
cudaMemcpyToSymbolAsync(const T &
# 776
symbol, const void *
# 777
src, size_t 
# 778
count, size_t 
# 779
offset = 0, cudaMemcpyKind 
# 780
kind = cudaMemcpyHostToDevice, cudaStream_t 
# 781
stream = 0) 
# 783
{ 
# 784
return ::cudaMemcpyToSymbolAsync((const void *)(&symbol), src, count, offset, kind, stream); 
# 785
} 
# 822 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/cuda_runtime.h"
template< class T> static inline cudaError_t 
# 823
cudaMemcpyFromSymbol(void *
# 824
dst, const T &
# 825
symbol, size_t 
# 826
count, size_t 
# 827
offset = 0, cudaMemcpyKind 
# 828
kind = cudaMemcpyDeviceToHost) 
# 830
{ 
# 831
return ::cudaMemcpyFromSymbol(dst, (const void *)(&symbol), count, offset, kind); 
# 832
} 
# 876 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/cuda_runtime.h"
template< class T> static inline cudaError_t 
# 877
cudaMemcpyFromSymbolAsync(void *
# 878
dst, const T &
# 879
symbol, size_t 
# 880
count, size_t 
# 881
offset = 0, cudaMemcpyKind 
# 882
kind = cudaMemcpyDeviceToHost, cudaStream_t 
# 883
stream = 0) 
# 885
{ 
# 886
return ::cudaMemcpyFromSymbolAsync(dst, (const void *)(&symbol), count, offset, kind, stream); 
# 887
} 
# 945 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/cuda_runtime.h"
template< class T> static inline cudaError_t 
# 946
cudaGraphAddMemcpyNodeToSymbol(cudaGraphNode_t *
# 947
pGraphNode, cudaGraph_t 
# 948
graph, const cudaGraphNode_t *
# 949
pDependencies, size_t 
# 950
numDependencies, const T &
# 951
symbol, const void *
# 952
src, size_t 
# 953
count, size_t 
# 954
offset, cudaMemcpyKind 
# 955
kind) 
# 956
{ 
# 957
return ::cudaGraphAddMemcpyNodeToSymbol(pGraphNode, graph, pDependencies, numDependencies, (const void *)(&symbol), src, count, offset, kind); 
# 958
} 
# 1016 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/cuda_runtime.h"
template< class T> static inline cudaError_t 
# 1017
cudaGraphAddMemcpyNodeFromSymbol(cudaGraphNode_t *
# 1018
pGraphNode, cudaGraph_t 
# 1019
graph, const cudaGraphNode_t *
# 1020
pDependencies, size_t 
# 1021
numDependencies, void *
# 1022
dst, const T &
# 1023
symbol, size_t 
# 1024
count, size_t 
# 1025
offset, cudaMemcpyKind 
# 1026
kind) 
# 1027
{ 
# 1028
return ::cudaGraphAddMemcpyNodeFromSymbol(pGraphNode, graph, pDependencies, numDependencies, dst, (const void *)(&symbol), count, offset, kind); 
# 1029
} 
# 1067 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/cuda_runtime.h"
template< class T> static inline cudaError_t 
# 1068
cudaGraphMemcpyNodeSetParamsToSymbol(cudaGraphNode_t 
# 1069
node, const T &
# 1070
symbol, const void *
# 1071
src, size_t 
# 1072
count, size_t 
# 1073
offset, cudaMemcpyKind 
# 1074
kind) 
# 1075
{ 
# 1076
return ::cudaGraphMemcpyNodeSetParamsToSymbol(node, (const void *)(&symbol), src, count, offset, kind); 
# 1077
} 
# 1115 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/cuda_runtime.h"
template< class T> static inline cudaError_t 
# 1116
cudaGraphMemcpyNodeSetParamsFromSymbol(cudaGraphNode_t 
# 1117
node, void *
# 1118
dst, const T &
# 1119
symbol, size_t 
# 1120
count, size_t 
# 1121
offset, cudaMemcpyKind 
# 1122
kind) 
# 1123
{ 
# 1124
return ::cudaGraphMemcpyNodeSetParamsFromSymbol(node, dst, (const void *)(&symbol), count, offset, kind); 
# 1125
} 
# 1173 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/cuda_runtime.h"
template< class T> static inline cudaError_t 
# 1174
cudaGraphExecMemcpyNodeSetParamsToSymbol(cudaGraphExec_t 
# 1175
hGraphExec, cudaGraphNode_t 
# 1176
node, const T &
# 1177
symbol, const void *
# 1178
src, size_t 
# 1179
count, size_t 
# 1180
offset, cudaMemcpyKind 
# 1181
kind) 
# 1182
{ 
# 1183
return ::cudaGraphExecMemcpyNodeSetParamsToSymbol(hGraphExec, node, (const void *)(&symbol), src, count, offset, kind); 
# 1184
} 
# 1232 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/cuda_runtime.h"
template< class T> static inline cudaError_t 
# 1233
cudaGraphExecMemcpyNodeSetParamsFromSymbol(cudaGraphExec_t 
# 1234
hGraphExec, cudaGraphNode_t 
# 1235
node, void *
# 1236
dst, const T &
# 1237
symbol, size_t 
# 1238
count, size_t 
# 1239
offset, cudaMemcpyKind 
# 1240
kind) 
# 1241
{ 
# 1242
return ::cudaGraphExecMemcpyNodeSetParamsFromSymbol(hGraphExec, node, dst, (const void *)(&symbol), count, offset, kind); 
# 1243
} 
# 1271 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/cuda_runtime.h"
template< class T> static inline cudaError_t 
# 1272
cudaUserObjectCreate(cudaUserObject_t *
# 1273
object_out, T *
# 1274
objectToWrap, unsigned 
# 1275
initialRefcount, unsigned 
# 1276
flags) 
# 1277
{ 
# 1278
return ::cudaUserObjectCreate(object_out, objectToWrap, [](void *
# 1281
vpObj) { delete (reinterpret_cast< T *>(vpObj)); } , initialRefcount, flags); 
# 1284
} 
# 1286
template< class T> static inline cudaError_t 
# 1287
cudaUserObjectCreate(cudaUserObject_t *
# 1288
object_out, T *
# 1289
objectToWrap, unsigned 
# 1290
initialRefcount, cudaUserObjectFlags 
# 1291
flags) 
# 1292
{ 
# 1293
return cudaUserObjectCreate(object_out, objectToWrap, initialRefcount, (unsigned)flags); 
# 1294
} 
# 1321 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/cuda_runtime.h"
template< class T> static inline cudaError_t 
# 1322
cudaGetSymbolAddress(void **
# 1323
devPtr, const T &
# 1324
symbol) 
# 1326
{ 
# 1327
return ::cudaGetSymbolAddress(devPtr, (const void *)(&symbol)); 
# 1328
} 
# 1353 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/cuda_runtime.h"
template< class T> static inline cudaError_t 
# 1354
cudaGetSymbolSize(size_t *
# 1355
size, const T &
# 1356
symbol) 
# 1358
{ 
# 1359
return ::cudaGetSymbolSize(size, (const void *)(&symbol)); 
# 1360
} 
# 1397 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/cuda_runtime.h"
template< class T, int dim, cudaTextureReadMode readMode> 
# 1398
__attribute((deprecated)) static inline cudaError_t cudaBindTexture(size_t *
# 1399
offset, const texture< T, dim, readMode>  &
# 1400
tex, const void *
# 1401
devPtr, const cudaChannelFormatDesc &
# 1402
desc, size_t 
# 1403
size = ((2147483647) * 2U) + 1U) 
# 1405 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/cuda_runtime.h"
{ 
# 1406
return ::cudaBindTexture(offset, &tex, devPtr, &desc, size); 
# 1407
} 
# 1443 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/cuda_runtime.h"
template< class T, int dim, cudaTextureReadMode readMode> 
# 1444
__attribute((deprecated)) static inline cudaError_t cudaBindTexture(size_t *
# 1445
offset, const texture< T, dim, readMode>  &
# 1446
tex, const void *
# 1447
devPtr, size_t 
# 1448
size = ((2147483647) * 2U) + 1U) 
# 1450 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/cuda_runtime.h"
{ 
# 1451
return cudaBindTexture(offset, tex, devPtr, (tex.channelDesc), size); 
# 1452
} 
# 1500 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/cuda_runtime.h"
template< class T, int dim, cudaTextureReadMode readMode> 
# 1501
__attribute((deprecated)) static inline cudaError_t cudaBindTexture2D(size_t *
# 1502
offset, const texture< T, dim, readMode>  &
# 1503
tex, const void *
# 1504
devPtr, const cudaChannelFormatDesc &
# 1505
desc, size_t 
# 1506
width, size_t 
# 1507
height, size_t 
# 1508
pitch) 
# 1510
{ 
# 1511
return ::cudaBindTexture2D(offset, &tex, devPtr, &desc, width, height, pitch); 
# 1512
} 
# 1559 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/cuda_runtime.h"
template< class T, int dim, cudaTextureReadMode readMode> 
# 1560
__attribute((deprecated)) static inline cudaError_t cudaBindTexture2D(size_t *
# 1561
offset, const texture< T, dim, readMode>  &
# 1562
tex, const void *
# 1563
devPtr, size_t 
# 1564
width, size_t 
# 1565
height, size_t 
# 1566
pitch) 
# 1568
{ 
# 1569
return ::cudaBindTexture2D(offset, &tex, devPtr, &(tex.channelDesc), width, height, pitch); 
# 1570
} 
# 1602 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/cuda_runtime.h"
template< class T, int dim, cudaTextureReadMode readMode> 
# 1603
__attribute((deprecated)) static inline cudaError_t cudaBindTextureToArray(const texture< T, dim, readMode>  &
# 1604
tex, cudaArray_const_t 
# 1605
array, const cudaChannelFormatDesc &
# 1606
desc) 
# 1608
{ 
# 1609
return ::cudaBindTextureToArray(&tex, array, &desc); 
# 1610
} 
# 1641 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/cuda_runtime.h"
template< class T, int dim, cudaTextureReadMode readMode> 
# 1642
__attribute((deprecated)) static inline cudaError_t cudaBindTextureToArray(const texture< T, dim, readMode>  &
# 1643
tex, cudaArray_const_t 
# 1644
array) 
# 1646
{ 
# 1647
cudaChannelFormatDesc desc; 
# 1648
cudaError_t err = ::cudaGetChannelDesc(&desc, array); 
# 1650
return (err == (cudaSuccess)) ? cudaBindTextureToArray(tex, array, desc) : err; 
# 1651
} 
# 1683 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/cuda_runtime.h"
template< class T, int dim, cudaTextureReadMode readMode> 
# 1684
__attribute((deprecated)) static inline cudaError_t cudaBindTextureToMipmappedArray(const texture< T, dim, readMode>  &
# 1685
tex, cudaMipmappedArray_const_t 
# 1686
mipmappedArray, const cudaChannelFormatDesc &
# 1687
desc) 
# 1689
{ 
# 1690
return ::cudaBindTextureToMipmappedArray(&tex, mipmappedArray, &desc); 
# 1691
} 
# 1722 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/cuda_runtime.h"
template< class T, int dim, cudaTextureReadMode readMode> 
# 1723
__attribute((deprecated)) static inline cudaError_t cudaBindTextureToMipmappedArray(const texture< T, dim, readMode>  &
# 1724
tex, cudaMipmappedArray_const_t 
# 1725
mipmappedArray) 
# 1727
{ 
# 1728
cudaChannelFormatDesc desc; 
# 1729
cudaArray_t levelArray; 
# 1730
cudaError_t err = ::cudaGetMipmappedArrayLevel(&levelArray, mipmappedArray, 0); 
# 1732
if (err != (cudaSuccess)) { 
# 1733
return err; 
# 1734
}  
# 1735
err = ::cudaGetChannelDesc(&desc, levelArray); 
# 1737
return (err == (cudaSuccess)) ? cudaBindTextureToMipmappedArray(tex, mipmappedArray, desc) : err; 
# 1738
} 
# 1765 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/cuda_runtime.h"
template< class T, int dim, cudaTextureReadMode readMode> 
# 1766
__attribute((deprecated)) static inline cudaError_t cudaUnbindTexture(const texture< T, dim, readMode>  &
# 1767
tex) 
# 1769
{ 
# 1770
return ::cudaUnbindTexture(&tex); 
# 1771
} 
# 1801 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/cuda_runtime.h"
template< class T, int dim, cudaTextureReadMode readMode> 
# 1802
__attribute((deprecated)) static inline cudaError_t cudaGetTextureAlignmentOffset(size_t *
# 1803
offset, const texture< T, dim, readMode>  &
# 1804
tex) 
# 1806
{ 
# 1807
return ::cudaGetTextureAlignmentOffset(offset, &tex); 
# 1808
} 
# 1853 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/cuda_runtime.h"
template< class T> static inline cudaError_t 
# 1854
cudaFuncSetCacheConfig(T *
# 1855
func, cudaFuncCache 
# 1856
cacheConfig) 
# 1858
{ 
# 1859
return ::cudaFuncSetCacheConfig((const void *)func, cacheConfig); 
# 1860
} 
# 1862
template< class T> static inline cudaError_t 
# 1863
cudaFuncSetSharedMemConfig(T *
# 1864
func, cudaSharedMemConfig 
# 1865
config) 
# 1867
{ 
# 1868
return ::cudaFuncSetSharedMemConfig((const void *)func, config); 
# 1869
} 
# 1901 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/cuda_runtime.h"
template< class T> inline cudaError_t 
# 1902
cudaOccupancyMaxActiveBlocksPerMultiprocessor(int *
# 1903
numBlocks, T 
# 1904
func, int 
# 1905
blockSize, size_t 
# 1906
dynamicSMemSize) 
# 1907
{ 
# 1908
return ::cudaOccupancyMaxActiveBlocksPerMultiprocessorWithFlags(numBlocks, (const void *)func, blockSize, dynamicSMemSize, 0); 
# 1909
} 
# 1953 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/cuda_runtime.h"
template< class T> inline cudaError_t 
# 1954
cudaOccupancyMaxActiveBlocksPerMultiprocessorWithFlags(int *
# 1955
numBlocks, T 
# 1956
func, int 
# 1957
blockSize, size_t 
# 1958
dynamicSMemSize, unsigned 
# 1959
flags) 
# 1960
{ 
# 1961
return ::cudaOccupancyMaxActiveBlocksPerMultiprocessorWithFlags(numBlocks, (const void *)func, blockSize, dynamicSMemSize, flags); 
# 1962
} 
# 1967
class __cudaOccupancyB2DHelper { 
# 1968
size_t n; 
# 1970
public: __cudaOccupancyB2DHelper(size_t n_) : n(n_) { } 
# 1971
size_t operator()(int) 
# 1972
{ 
# 1973
return n; 
# 1974
} 
# 1975
}; 
# 2023 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/cuda_runtime.h"
template< class UnaryFunction, class T> static inline cudaError_t 
# 2024
cudaOccupancyMaxPotentialBlockSizeVariableSMemWithFlags(int *
# 2025
minGridSize, int *
# 2026
blockSize, T 
# 2027
func, UnaryFunction 
# 2028
blockSizeToDynamicSMemSize, int 
# 2029
blockSizeLimit = 0, unsigned 
# 2030
flags = 0) 
# 2031
{ 
# 2032
cudaError_t status; 
# 2035
int device; 
# 2036
cudaFuncAttributes attr; 
# 2039
int maxThreadsPerMultiProcessor; 
# 2040
int warpSize; 
# 2041
int devMaxThreadsPerBlock; 
# 2042
int multiProcessorCount; 
# 2043
int funcMaxThreadsPerBlock; 
# 2044
int occupancyLimit; 
# 2045
int granularity; 
# 2048
int maxBlockSize = 0; 
# 2049
int numBlocks = 0; 
# 2050
int maxOccupancy = 0; 
# 2053
int blockSizeToTryAligned; 
# 2054
int blockSizeToTry; 
# 2055
int blockSizeLimitAligned; 
# 2056
int occupancyInBlocks; 
# 2057
int occupancyInThreads; 
# 2058
size_t dynamicSMemSize; 
# 2064
if (((!minGridSize) || (!blockSize)) || (!func)) { 
# 2065
return cudaErrorInvalidValue; 
# 2066
}  
# 2072
status = ::cudaGetDevice(&device); 
# 2073
if (status != (cudaSuccess)) { 
# 2074
return status; 
# 2075
}  
# 2077
status = cudaDeviceGetAttribute(&maxThreadsPerMultiProcessor, cudaDevAttrMaxThreadsPerMultiProcessor, device); 
# 2081
if (status != (cudaSuccess)) { 
# 2082
return status; 
# 2083
}  
# 2085
status = cudaDeviceGetAttribute(&warpSize, cudaDevAttrWarpSize, device); 
# 2089
if (status != (cudaSuccess)) { 
# 2090
return status; 
# 2091
}  
# 2093
status = cudaDeviceGetAttribute(&devMaxThreadsPerBlock, cudaDevAttrMaxThreadsPerBlock, device); 
# 2097
if (status != (cudaSuccess)) { 
# 2098
return status; 
# 2099
}  
# 2101
status = cudaDeviceGetAttribute(&multiProcessorCount, cudaDevAttrMultiProcessorCount, device); 
# 2105
if (status != (cudaSuccess)) { 
# 2106
return status; 
# 2107
}  
# 2109
status = cudaFuncGetAttributes(&attr, func); 
# 2110
if (status != (cudaSuccess)) { 
# 2111
return status; 
# 2112
}  
# 2114
funcMaxThreadsPerBlock = (attr.maxThreadsPerBlock); 
# 2120
occupancyLimit = maxThreadsPerMultiProcessor; 
# 2121
granularity = warpSize; 
# 2123
if (blockSizeLimit == 0) { 
# 2124
blockSizeLimit = devMaxThreadsPerBlock; 
# 2125
}  
# 2127
if (devMaxThreadsPerBlock < blockSizeLimit) { 
# 2128
blockSizeLimit = devMaxThreadsPerBlock; 
# 2129
}  
# 2131
if (funcMaxThreadsPerBlock < blockSizeLimit) { 
# 2132
blockSizeLimit = funcMaxThreadsPerBlock; 
# 2133
}  
# 2135
blockSizeLimitAligned = (((blockSizeLimit + (granularity - 1)) / granularity) * granularity); 
# 2137
for (blockSizeToTryAligned = blockSizeLimitAligned; blockSizeToTryAligned > 0; blockSizeToTryAligned -= granularity) { 
# 2141
if (blockSizeLimit < blockSizeToTryAligned) { 
# 2142
blockSizeToTry = blockSizeLimit; 
# 2143
} else { 
# 2144
blockSizeToTry = blockSizeToTryAligned; 
# 2145
}  
# 2147
dynamicSMemSize = blockSizeToDynamicSMemSize(blockSizeToTry); 
# 2149
status = cudaOccupancyMaxActiveBlocksPerMultiprocessorWithFlags(&occupancyInBlocks, func, blockSizeToTry, dynamicSMemSize, flags); 
# 2156
if (status != (cudaSuccess)) { 
# 2157
return status; 
# 2158
}  
# 2160
occupancyInThreads = (blockSizeToTry * occupancyInBlocks); 
# 2162
if (occupancyInThreads > maxOccupancy) { 
# 2163
maxBlockSize = blockSizeToTry; 
# 2164
numBlocks = occupancyInBlocks; 
# 2165
maxOccupancy = occupancyInThreads; 
# 2166
}  
# 2170
if (occupancyLimit == maxOccupancy) { 
# 2171
break; 
# 2172
}  
# 2173
}  
# 2181
(*minGridSize) = (numBlocks * multiProcessorCount); 
# 2182
(*blockSize) = maxBlockSize; 
# 2184
return status; 
# 2185
} 
# 2219 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/cuda_runtime.h"
template< class UnaryFunction, class T> static inline cudaError_t 
# 2220
cudaOccupancyMaxPotentialBlockSizeVariableSMem(int *
# 2221
minGridSize, int *
# 2222
blockSize, T 
# 2223
func, UnaryFunction 
# 2224
blockSizeToDynamicSMemSize, int 
# 2225
blockSizeLimit = 0) 
# 2226
{ 
# 2227
return cudaOccupancyMaxPotentialBlockSizeVariableSMemWithFlags(minGridSize, blockSize, func, blockSizeToDynamicSMemSize, blockSizeLimit, 0); 
# 2228
} 
# 2265 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/cuda_runtime.h"
template< class T> static inline cudaError_t 
# 2266
cudaOccupancyMaxPotentialBlockSize(int *
# 2267
minGridSize, int *
# 2268
blockSize, T 
# 2269
func, size_t 
# 2270
dynamicSMemSize = 0, int 
# 2271
blockSizeLimit = 0) 
# 2272
{ 
# 2273
return cudaOccupancyMaxPotentialBlockSizeVariableSMemWithFlags(minGridSize, blockSize, func, ((__cudaOccupancyB2DHelper)(dynamicSMemSize)), blockSizeLimit, 0); 
# 2274
} 
# 2303 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/cuda_runtime.h"
template< class T> static inline cudaError_t 
# 2304
cudaOccupancyAvailableDynamicSMemPerBlock(size_t *
# 2305
dynamicSmemSize, T 
# 2306
func, int 
# 2307
numBlocks, int 
# 2308
blockSize) 
# 2309
{ 
# 2310
return ::cudaOccupancyAvailableDynamicSMemPerBlock(dynamicSmemSize, (const void *)func, numBlocks, blockSize); 
# 2311
} 
# 2362 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/cuda_runtime.h"
template< class T> static inline cudaError_t 
# 2363
cudaOccupancyMaxPotentialBlockSizeWithFlags(int *
# 2364
minGridSize, int *
# 2365
blockSize, T 
# 2366
func, size_t 
# 2367
dynamicSMemSize = 0, int 
# 2368
blockSizeLimit = 0, unsigned 
# 2369
flags = 0) 
# 2370
{ 
# 2371
return cudaOccupancyMaxPotentialBlockSizeVariableSMemWithFlags(minGridSize, blockSize, func, ((__cudaOccupancyB2DHelper)(dynamicSMemSize)), blockSizeLimit, flags); 
# 2372
} 
# 2405 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/cuda_runtime.h"
template< class T> inline cudaError_t 
# 2406
cudaFuncGetAttributes(cudaFuncAttributes *
# 2407
attr, T *
# 2408
entry) 
# 2410
{ 
# 2411
return ::cudaFuncGetAttributes(attr, (const void *)entry); 
# 2412
} 
# 2450 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/cuda_runtime.h"
template< class T> static inline cudaError_t 
# 2451
cudaFuncSetAttribute(T *
# 2452
entry, cudaFuncAttribute 
# 2453
attr, int 
# 2454
value) 
# 2456
{ 
# 2457
return ::cudaFuncSetAttribute((const void *)entry, attr, value); 
# 2458
} 
# 2482 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/cuda_runtime.h"
template< class T, int dim> 
# 2483
__attribute((deprecated)) static inline cudaError_t cudaBindSurfaceToArray(const surface< T, dim>  &
# 2484
surf, cudaArray_const_t 
# 2485
array, const cudaChannelFormatDesc &
# 2486
desc) 
# 2488
{ 
# 2489
return ::cudaBindSurfaceToArray(&surf, array, &desc); 
# 2490
} 
# 2513 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/cuda_runtime.h"
template< class T, int dim> 
# 2514
__attribute((deprecated)) static inline cudaError_t cudaBindSurfaceToArray(const surface< T, dim>  &
# 2515
surf, cudaArray_const_t 
# 2516
array) 
# 2518
{ 
# 2519
cudaChannelFormatDesc desc; 
# 2520
cudaError_t err = ::cudaGetChannelDesc(&desc, array); 
# 2522
return (err == (cudaSuccess)) ? cudaBindSurfaceToArray(surf, array, desc) : err; 
# 2523
} 
# 2534 "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/include/cuda_runtime.h"
#pragma GCC diagnostic pop
# 40 "CMakeCUDACompilerId.cu"
const char *info_compiler = ("INFO:compiler[NVIDIA]"); 
# 257 "CMakeCUDACompilerId.cu"
const char info_version[] = {'I', 'N', 'F', 'O', ':', 'c', 'o', 'm', 'p', 'i', 'l', 'e', 'r', '_', 'v', 'e', 'r', 's', 'i', 'o', 'n', '[', (('0') + ((11 / 10000000) % 10)), (('0') + ((11 / 1000000) % 10)), (('0') + ((11 / 100000) % 10)), (('0') + ((11 / 10000) % 10)), (('0') + ((11 / 1000) % 10)), (('0') + ((11 / 100) % 10)), (('0') + ((11 / 10) % 10)), (('0') + (11 % 10)), '.', (('0') + ((4 / 10000000) % 10)), (('0') + ((4 / 1000000) % 10)), (('0') + ((4 / 100000) % 10)), (('0') + ((4 / 10000) % 10)), (('0') + ((4 / 1000) % 10)), (('0') + ((4 / 100) % 10)), (('0') + ((4 / 10) % 10)), (('0') + (4 % 10)), '.', (('0') + ((48 / 10000000) % 10)), (('0') + ((48 / 1000000) % 10)), (('0') + ((48 / 100000) % 10)), (('0') + ((48 / 10000) % 10)), (('0') + ((48 / 1000) % 10)), (('0') + ((48 / 100) % 10)), (('0') + ((48 / 10) % 10)), (('0') + (48 % 10)), ']', '\000'}; 
# 304 "CMakeCUDACompilerId.cu"
const char *info_platform = ("INFO:platform[Linux]"); 
# 305
const char *info_arch = ("INFO:arch[]"); 
# 310
const char *info_language_dialect_default = ("INFO:dialect_default[14]"); 
# 326
int main(int argc, char *argv[]) 
# 327
{ 
# 328
int require = 0; 
# 329
require += (info_compiler[argc]); 
# 330
require += (info_platform[argc]); 
# 332
require += (info_version[argc]); 
# 340
require += (info_language_dialect_default[argc]); 
# 341
(void)argv; 
# 342
return require; 
# 343
} 

# 1 "CMakeCUDACompilerId.cudafe1.stub.c"
#define _NV_ANON_NAMESPACE _GLOBAL__N__27_CMakeCUDACompilerId_cpp1_ii_bd57c623
#ifdef _NV_ANON_NAMESPACE
#endif
# 1 "CMakeCUDACompilerId.cudafe1.stub.c"
#include "CMakeCUDACompilerId.cudafe1.stub.c"
# 1 "CMakeCUDACompilerId.cudafe1.stub.c"
#undef _NV_ANON_NAMESPACE
