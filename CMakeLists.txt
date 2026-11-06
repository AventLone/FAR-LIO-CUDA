cmake_minimum_required(VERSION 3.28)
if (NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES native)
endif ()
project(localization LANGUAGES C CXX CUDA)

if (NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif ()

if (CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
    add_compile_options(
        $<$<COMPILE_LANGUAGE:CXX>:-Wall>
        $<$<COMPILE_LANGUAGE:CXX>:-Wextra>
        $<$<COMPILE_LANGUAGE:CXX>:-Wpedantic>
    )
endif ()

# --------------------------------------------------------------------------------
# Find ROS2 dependencies
# --------------------------------------------------------------------------------
find_package(ament_cmake_auto REQUIRED)
ament_auto_find_build_dependencies() # Automaticly find all the packages mentioned in 'package.xml'

file(GLOB_RECURSE SOURCE_FILES CONFIGURE_DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/source/*.cpp)

find_package(Sophus REQUIRED)
find_package(TBB REQUIRED)
find_package(CUDAToolkit REQUIRED)
find_package(CCCL REQUIRED CONFIG HINTS
    "${CUDAToolkit_LIBRARY_ROOT}/lib64/cmake/cccl"
    "${CUDAToolkit_LIBRARY_ROOT}/targets/${CMAKE_SYSTEM_PROCESSOR}-linux/lib/cmake/cccl"
)
find_package(PCL REQUIRED COMPONENTS common)

include(FetchContent)
FetchContent_Declare(cuco_headers
    GIT_REPOSITORY https://github.com/NVIDIA/cuCollections.git
    # Includes the CCCL 3 / CUDA 13 compatibility fix for thrust::identity.
    GIT_TAG 7b422c00f3541e2472e8d1047b7c3bb4e83c7e2c
    # Only use the headers; CCCL is supplied by the CUDA toolkit.
    SOURCE_SUBDIR include
)
FetchContent_MakeAvailable(cuco_headers)

add_library(sparse_voxel STATIC source/types/SparseVoxel.cu)
target_include_directories(sparse_voxel PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
)
target_include_directories(sparse_voxel SYSTEM PUBLIC ${PCL_INCLUDE_DIRS})
target_include_directories(sparse_voxel SYSTEM PRIVATE ${cuco_headers_SOURCE_DIR}/include)
target_link_libraries(sparse_voxel PUBLIC pcl_common CUDA::cudart CCCL::CCCL)
target_compile_features(sparse_voxel PUBLIC cxx_std_20 PRIVATE cuda_std_20)
target_compile_options(sparse_voxel PRIVATE --expt-relaxed-constexpr --extended-lambda --fmad=false)
set_target_properties(sparse_voxel PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

add_library(gicp STATIC source/GICP/GICP.cu)
target_link_libraries(gicp PUBLIC sparse_voxel Sophus::Sophus)
target_compile_features(gicp PRIVATE cuda_std_20)
target_compile_options(gicp PRIVATE --expt-relaxed-constexpr --extended-lambda --fmad=false)
set_target_properties(gicp PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

ament_auto_add_executable(localization ${SOURCE_FILES})
target_include_directories(localization PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include/${PROJECT_NAME}>
)
target_link_libraries(localization Sophus::Sophus TBB::tbb gicp)
target_compile_features(localization PUBLIC c_std_99 cxx_std_23)

if (BUILD_TESTING)
    add_executable(sparse_voxel_gpu_test tests/sparse_voxel_gpu_test.cu)
    target_link_libraries(sparse_voxel_gpu_test PRIVATE sparse_voxel)
    target_compile_features(sparse_voxel_gpu_test PRIVATE cuda_std_20)
    target_compile_options(sparse_voxel_gpu_test PRIVATE --expt-relaxed-constexpr --extended-lambda --fmad=false)
    set_target_properties(sparse_voxel_gpu_test PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
    add_test(NAME sparse_voxel_gpu_test COMMAND sparse_voxel_gpu_test)

    add_executable(gicp_cuda_test tests/gicp_cuda_test.cu)
    target_link_libraries(gicp_cuda_test PRIVATE gicp)
    target_compile_features(gicp_cuda_test PRIVATE cuda_std_20)
    target_compile_options(gicp_cuda_test PRIVATE --expt-relaxed-constexpr --extended-lambda --fmad=false)
    set_target_properties(gicp_cuda_test PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
    add_test(NAME gicp_cuda_test COMMAND gicp_cuda_test)

    find_package(ament_lint_auto REQUIRED)
    set(ament_cmake_copyright_FOUND TRUE)
    set(ament_cmake_cpplint_FOUND TRUE)
    ament_lint_auto_find_test_dependencies()
endif ()

# --------------------------------------------------------------------------------
# Install
# --------------------------------------------------------------------------------
ament_auto_package(INSTALL_TO_SHARE)