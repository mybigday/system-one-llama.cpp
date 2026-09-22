add_library(llama-system-one STATIC system-one.cpp system-one.h)
target_link_libraries(llama-system-one PUBLIC llama-common llama ${CMAKE_THREAD_LIBS_INIT})
target_include_directories(llama-system-one PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

# it is linked into libllama-server-impl.so, so it has to be position independent
if (BUILD_SHARED_LIBS)
    set_target_properties(llama-system-one PROPERTIES POSITION_INDEPENDENT_CODE ON)
endif()
target_compile_features(llama-system-one PRIVATE cxx_std_17)

set(TARGET llama-system-one-tok-check)
add_executable(${TARGET} tok-check.cpp)
target_link_libraries(${TARGET} PRIVATE llama-system-one llama-common llama ${CMAKE_THREAD_LIBS_INIT})
target_include_directories(${TARGET} PRIVATE ${CMAKE_SOURCE_DIR}/vendor)
target_compile_features(${TARGET} PRIVATE cxx_std_17)

set(TARGET llama-system-one-e2e-check)
add_executable(${TARGET} e2e-check.cpp)
target_link_libraries(${TARGET} PRIVATE llama-system-one llama-common llama ggml ${CMAKE_THREAD_LIBS_INIT})
target_include_directories(${TARGET} PRIVATE ${CMAKE_SOURCE_DIR}/vendor)
target_compile_features(${TARGET} PRIVATE cxx_std_17)
