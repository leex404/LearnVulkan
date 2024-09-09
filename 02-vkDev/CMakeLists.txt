
set(EXAMPLES
	02-draw_triangle_with_vertex_buffer
)


add_executable(${EXAMPLES} 
	src/main.cpp
)
#target_link_libraries(${EXAMPLES} ${Vulkan_LIBRARY} ${WINLIBS})