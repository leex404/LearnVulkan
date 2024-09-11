
set(EXAMPLES
	06-draw_model
)


add_executable(${EXAMPLES} 
	src/main.cpp
)
#target_link_libraries(${EXAMPLES} ${Vulkan_LIBRARY} ${WINLIBS})