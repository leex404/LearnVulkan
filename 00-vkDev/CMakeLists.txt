
set(EXAMPLES
	00-empty_window
)


add_executable(${EXAMPLES} 
	src/main.cpp
)
#target_link_libraries(${EXAMPLES} ${Vulkan_LIBRARY} ${WINLIBS})