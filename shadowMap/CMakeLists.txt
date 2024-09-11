
set(EXAMPLES
	shadowMap_demo
)


add_executable(${EXAMPLES} 
	src/main.cpp
)
#target_link_libraries(${EXAMPLES} ${Vulkan_LIBRARY} ${WINLIBS})