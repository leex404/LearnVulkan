cmake_minimum_required (VERSION 3.5)

project (LearnVulkan)

link_directories( ${CMAKE_SOURCE_DIR}/lib )

set(CMAKE_LIBRARY_PATH ${CMAKE_SOURCE_DIR}/lib)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/bin)


set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

message(STATUS "Using module to find Vulkan")
find_package(Vulkan)

if(NOT Vulkan_FOUND)
	message(FATAL_ERROR "Could not find Vulkan library!")
else()
	message(STATUS "Vulkan include dir: ${Vulkan_INCLUDE_DIRS}")
	include_directories(${Vulkan_INCLUDE_DIRS})
	message(STATUS "Vulkan library dir: ${Vulkan_LIBRARY}")
endif()

find_library( GLFW3_LIBRARY glfw3 "glfw3 1.4 library" )

include_directories( 
    ${CMAKE_SOURCE_DIR}/include 
	${Vulkan_INCLUDE_DIRS}
)


link_libraries( 
	${GLFW3_LIBRARY}
	${Vulkan_LIBRARY}
)


add_subdirectory(00-vkDev)
add_subdirectory(01-vkDev)
add_subdirectory(02-vkDev)
add_subdirectory(03-vkDev)
