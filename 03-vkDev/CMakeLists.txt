
set(EXAMPLES
	03-draw_triangle_with_uniform
)


add_executable(${EXAMPLES} 
	src/main.cpp
)
#target_link_libraries(${EXAMPLES} ${Vulkan_LIBRARY} ${WINLIBS})