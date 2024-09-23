#define VK_USE_PLATFORM_WIN32_KHR
#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>
#define GLFW_EXPOSE_NATIVE_WIN32
#include <GLFW/glfw3native.h>

#define GLM_FORCE_RADIANS
#define GLM_ENABLE_EXPERIMENTAL
#define GLM_FORCE_DEPTH_ZERO_TO_ONE 
#define GLM_FORCE_DEFAULT_ALIGNED_GENTYPES
#include <glm/glm.hpp>
#include <glm/gtx/hash.hpp>
#include <glm/gtc/matrix_transform.hpp>

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image/stb_image.h"

#define TINYOBJLOADER_IMPLEMENTATION
#include <tinyobjloader/tiny_obj_loader.h>

#include <vulkan/vulkan.h>

#include <set>
#include <array>
#include <limits>
#include <vector>
#include <chrono>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <optional>
#include <stdexcept>
#include <algorithm>
#include <unordered_map>

const int WIDTH = 1280;
const int HEIGHT = 1280;

const uint32_t SHADOW_SIZE = 2048;

const int MAX_FRAMES_IN_FLIGHT = 2;

const std::string title = "Vulkan - ShadowMap Demo";
const std::string vertShaderPath = "../../resources/shaders/shadowMap/vert.spv";
const std::string fragShaderPath = "../../resources/shaders/shadowMap/frag.spv";
const std::string shadowVertShaderPath = "../../resources/shaders/shadowMap/shadow_vert.spv";
const std::string shadowFragShaderPath = "../../resources/shaders/shadowMap/shadow_frag.spv";

const std::string roomModelPath = "../../resources/models/Marry.obj";
const std::string roomTexturePath = "../../resources/texture/MC003_Kozakura_Mari.png";


const std::vector<const char*> validationLayers = {
	"VK_LAYER_KHRONOS_validation"
};

const std::vector<const char*> deviceExtensions = {
	VK_KHR_SWAPCHAIN_EXTENSION_NAME
};

//#if NDEBUG
//const bool enbleValidationLayers = false;
//#else
//const bool enbleValidationLayers = true;
//#endif // NDEBUG

const bool enbleValidationLayers = true;

struct ShadowUBO
{
	glm::mat4 view;
	glm::mat4 proj;
};

struct UniformBufferObject
{
	glm::mat4 view;
	glm::mat4 proj;
	glm::mat4 lightVP;
};


struct ViewUniformBufferObject
{
	glm::vec4 cameraPos;
	glm::vec4 lightPos;
};

struct PushConstantData
{
	glm::mat4 modelMatrix;
} ;


struct Vertex
{
	glm::vec3 pos;
	glm::vec3 color;
	glm::vec3 normal;
	glm::vec2 texCoord;

	static VkVertexInputBindingDescription getBindingDescription()
	{
		VkVertexInputBindingDescription bindingDescription{};
		bindingDescription.binding = 0;
		bindingDescription.stride = sizeof(Vertex);
		bindingDescription.inputRate = VK_VERTEX_INPUT_RATE_VERTEX;

		return bindingDescription;
	}

	static std::array<VkVertexInputAttributeDescription, 4> getAttributeDescription()
	{
		std::array<VkVertexInputAttributeDescription, 4> attributeDescriptions{};
		// vertex
		attributeDescriptions[0].binding = 0;
		attributeDescriptions[0].location = 0;
		attributeDescriptions[0].format = VK_FORMAT_R32G32B32_SFLOAT;
		attributeDescriptions[0].offset = offsetof(Vertex, pos);

		// color
		attributeDescriptions[1].binding = 0;
		attributeDescriptions[1].location = 1;
		attributeDescriptions[1].format = VK_FORMAT_R32G32B32_SFLOAT;
		attributeDescriptions[1].offset = offsetof(Vertex, color);

		// normal
		attributeDescriptions[2].binding = 0;
		attributeDescriptions[2].location = 2;
		attributeDescriptions[2].format = VK_FORMAT_R32G32B32_SFLOAT;
		attributeDescriptions[2].offset = offsetof(Vertex, normal);

		// texture cooordinate
		attributeDescriptions[3].binding = 0;
		attributeDescriptions[3].location = 3;
		attributeDescriptions[3].format = VK_FORMAT_R32G32_SFLOAT;
		attributeDescriptions[3].offset = offsetof(Vertex, texCoord);

		return attributeDescriptions;
	}

	bool operator==(const Vertex& other) const
	{
		return pos == other.pos && color == other.color && normal == other.normal && texCoord == other.texCoord;
	}
};

struct Input
{
	glm::vec3 lightPos = { -5.0f, 5.0f, 3.0f };
	//glm::vec3 lightPos = { 1.0f, 1.0f, 9.0f };
	glm::vec3 cameraPos = { 0.0f, 2.0f, 10.0f };
	glm::vec3 cameraUp = { 0.0f, 1.0f, 0.0f };
	glm::vec3 target = { 0.0f, 0.0f, 0.0f };

	float zNear = 0.1f;
	float zFar = 15.0;

	float lightFOV = 45.0f;

}gInput;

struct RenderObject
{
	// vertex 
	std::vector<Vertex> vertices;
	VkBuffer vertexBuffer;
	VkDeviceMemory vertexBufferMemory;

	// index
	std::vector<uint32_t> indices;
	VkBuffer indexBuffer;
	VkDeviceMemory indexBufferMemory;

	// texture
	std::vector<VkImage> textureImages;
	std::vector<VkImageView> textureImageViews;
	std::vector<VkDeviceMemory> textureImageMemorys;

	std::vector<VkSampler> textureSamplers;

	// descriptor
	VkDescriptorPool descriptorPool;
	std::vector<VkDescriptorSet> descriptorSets;

	glm::mat4 modelMatrix = glm::mat4(1.0f);
};


struct BaseScenePass
{
	std::vector<RenderObject> renderObjects;   // models need render

	VkDescriptorSetLayout descriptorSetLayout;

	VkPipelineLayout pipelineLayout;

	VkPipeline pipeline;  // each model need a different pipeline ??
};

struct ShadowMapPass
{
	uint32_t width;
	uint32_t height;

	VkImage shadowMapImage;
	VkImageView shadowMapImageView;
	VkDeviceMemory shadowMapImageMemory;
	VkSampler shadowMapSampler;


	VkRenderPass renderPass;
	VkFramebuffer frameBuffer;

	VkPipeline pipeline;
	VkPipelineLayout pipelineLayout;

	VkDescriptorPool descriptorPool;
	VkDescriptorSetLayout descriptoSetLayout;
	std::vector<VkDescriptorSet> descriptoSets;

	std::vector<VkBuffer> uniformBuffers;
	std::vector<VkDeviceMemory> uniformBuffersMemory;
	std::vector<void*> uniformBuffersMapped;
};



namespace std {
	template<>
	struct hash<Vertex>
	{
		size_t operator()(Vertex const& vertex) const {
			return ((hash<glm::vec3>()(vertex.pos) ^
				(hash<glm::vec3>()(vertex.color) << 1)) >> 1) ^
				(hash<glm::vec2>()(vertex.texCoord) << 1);
		}
	};
}


static std::vector<char> readFile(const std::string& filename)
{
	std::ifstream file(filename, std::ios::ate | std::ios::binary);
	if (!file.is_open())
	{
		throw std::runtime_error("failed to open file!");
	}

	size_t fileSize = (size_t)file.tellg();
	std::vector<char> buffer(fileSize);

	file.seekg(0);
	file.read(buffer.data(), fileSize);
	file.close();
	return buffer;
}


VkResult CreateDebugUtilsMessengerEXT(
	VkInstance instance, 
	const VkDebugUtilsMessengerCreateInfoEXT* pCreateInfo, 
	const VkAllocationCallbacks* pAllocator, 
	VkDebugUtilsMessengerEXT* pDebugMessenger
)
{
	auto func = (PFN_vkCreateDebugUtilsMessengerEXT)vkGetInstanceProcAddr(instance, "vkCreateDebugUtilsMessengerEXT");
	if (func != nullptr) {
		return func(instance, pCreateInfo, pAllocator, pDebugMessenger);
	}
	return VK_ERROR_EXTENSION_NOT_PRESENT;
}

void DestroyDebugUtilsMessengerEXT(
	VkInstance instance, 
	VkDebugUtilsMessengerEXT debugMessenger, 
	const VkAllocationCallbacks* pAllocator
) 
{
	auto func = (PFN_vkDestroyDebugUtilsMessengerEXT)vkGetInstanceProcAddr(instance, "vkDestroyDebugUtilsMessengerEXT");
	if (func != nullptr) {
		func(instance, debugMessenger, pAllocator);
	}
}

struct QueueFamilyIndices
{
	std::optional<uint32_t> graphicsFamily;
	std::optional<uint32_t> presentFamily;

	bool isComplete()
	{
		return graphicsFamily.has_value() && presentFamily.has_value();
	}
};

struct SwapChainSupportDetails
{
	VkSurfaceCapabilitiesKHR capabilities;
	std::vector<VkSurfaceFormatKHR> formats;
	std::vector<VkPresentModeKHR> presentModes;
};

class HelloTriangle
{
public:
	void run()
	{
		initWindow();
		initVulkan();
		mainLoop();
		cleanup();
	}

private:
	void initWindow()
	{
		glfwInit();
		glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
		glfwWindowHint(GLFW_RESIZABLE, GLFW_TRUE);

		window = glfwCreateWindow(WIDTH, HEIGHT, title.c_str(), nullptr, nullptr);

		glfwSetWindowUserPointer(window, this);
		glfwSetFramebufferSizeCallback(window, framebufferResizeCallback);
	}

	void initVulkan()
	{
		createInstance();
		setupDebugMessenger();
		createSurface();
		pickPhysicalDevice();
		createLogicalDevice();
		creataSwapChain();
		createSwapChainImageViews();
		createRenderPass();

		createDepthResources();
		createFramebuffers();
		createCommandPool();
		createUniformBuffers();

		// model, texture pipline decriptor, layout pool set 
		createShadowMapPass();
		createSenceRenderPass();

		createCommandBuffer();
		createSyncObjects();
	}

	void mainLoop()
	{
		while (!glfwWindowShouldClose(window))
		{
			glfwPollEvents();

			drawFrame();
		}
		vkDeviceWaitIdle(device);
	}

	void createInstance()
	{
		if (enbleValidationLayers && !checkValidationLayerSupport())
		{
			throw std::runtime_error("validation layers requested, but not available!");
		}
		VkApplicationInfo appInfo{};
		appInfo.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
		appInfo.pApplicationName = "Hello Triangle";
		appInfo.applicationVersion = VK_MAKE_VERSION(1, 0, 0);
		appInfo.pEngineName = "No Engine";
		appInfo.engineVersion = VK_MAKE_VERSION(1, 0, 0);
		appInfo.apiVersion = VK_API_VERSION_1_0;

		VkInstanceCreateInfo createInfo{};
		createInfo.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
		createInfo.pApplicationInfo = &appInfo;

		// request extension need 
		std::vector<const char*> extensions = getRequiredExtensions();
		createInfo.enabledExtensionCount = static_cast<uint32_t>(extensions.size());
		createInfo.ppEnabledExtensionNames = extensions.data();

		// validation and debug messenger
		VkDebugUtilsMessengerCreateInfoEXT debugCreateInfo{};
		if (enbleValidationLayers)
		{
			createInfo.enabledLayerCount = static_cast<uint32_t>(validationLayers.size());
			createInfo.ppEnabledLayerNames = validationLayers.data();

			populateDebugMessengerCreateInfo(debugCreateInfo);
			createInfo.pNext = (VkDebugUtilsMessengerCreateInfoEXT*)&debugCreateInfo;
		}
		else
		{
			createInfo.enabledLayerCount = 0;
		}

		// checking for extension support
		uint32_t extensionCount = 0;
		vkEnumerateInstanceExtensionProperties(nullptr, &extensionCount, nullptr);

		std::vector<VkExtensionProperties> avaExtensions(extensionCount);
		vkEnumerateInstanceExtensionProperties(nullptr, &extensionCount, avaExtensions.data());
		std::cout << "avaliable extensions:\n";
		for (const auto& extension : avaExtensions)
		{
			std::cout << '\t' << extension.extensionName << '\n';
		}

		std::cout << "using extensions:\n";
		for (const auto& extension : extensions)
		{
			std::cout << '\t' << extension << '\n';
		}

		// create instance
		VkResult result = vkCreateInstance(&createInfo, nullptr, &instance);

		if (result != VK_SUCCESS)
		{
			throw std::runtime_error("failed to create instance");
		}
	}

	bool checkValidationLayerSupport()
	{
		uint32_t layerCount;
		vkEnumerateInstanceLayerProperties(&layerCount, nullptr);

		std::vector<VkLayerProperties> availableLayers(layerCount);
		vkEnumerateInstanceLayerProperties(&layerCount, availableLayers.data());

		// check if all validationLayers in availableLayers
		std::set<std::string> avaLayers;
		for (const auto& layer : availableLayers)
		{
			avaLayers.insert(layer.layerName);
		}

		for (const auto& layerName : validationLayers)
		{
			if (avaLayers.count(layerName) == 0)
			{
				return false;
			}
		}
		return true;
	}

	std::vector<const char*> getRequiredExtensions()
	{
		uint32_t glfwExtensionCount = 0;
		const char** glfwExtensions;

		glfwExtensions = glfwGetRequiredInstanceExtensions(&glfwExtensionCount);

		std::vector<const char*> extensions(glfwExtensions, glfwExtensions + glfwExtensionCount);

		if (enbleValidationLayers)
		{
			extensions.push_back(VK_EXT_DEBUG_UTILS_EXTENSION_NAME);
		}
		return extensions;
	}
	
	static VKAPI_ATTR VkBool32 VKAPI_CALL debugCallback(
		VkDebugUtilsMessageSeverityFlagBitsEXT messageSeverity,
		VkDebugUtilsMessageTypeFlagsEXT messageType,
		const VkDebugUtilsMessengerCallbackDataEXT* pCallbackData,
		void* pUserData
	)
	{
		std::cerr << "validation layer: " << pCallbackData->pMessage << '\n';
		return VK_FALSE;
	}
	void populateDebugMessengerCreateInfo(VkDebugUtilsMessengerCreateInfoEXT& createInfo)
	{
		createInfo.sType = VK_STRUCTURE_TYPE_DEBUG_UTILS_MESSENGER_CREATE_INFO_EXT;
		createInfo.messageSeverity = VK_DEBUG_UTILS_MESSAGE_SEVERITY_VERBOSE_BIT_EXT
			| VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT
			| VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT;
		createInfo.messageType = VK_DEBUG_UTILS_MESSAGE_TYPE_GENERAL_BIT_EXT
			| VK_DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT
			| VK_DEBUG_UTILS_MESSAGE_TYPE_PERFORMANCE_BIT_EXT;
		createInfo.pfnUserCallback = debugCallback;
		createInfo.pUserData = nullptr;
	}
	void setupDebugMessenger()
	{
		if (!enbleValidationLayers) {
			return;
		}

		VkDebugUtilsMessengerCreateInfoEXT createInfo{};
		populateDebugMessengerCreateInfo(createInfo);

		if (CreateDebugUtilsMessengerEXT(instance, &createInfo, nullptr, &debugMessenger) != VK_SUCCESS) {
			throw std::runtime_error("failed to set up debug messenger!");
		}
	}

	void createSurface()
	{
		if (glfwCreateWindowSurface(instance, window, nullptr, &surface) != VK_SUCCESS)
		{
			throw std::runtime_error("failed to create window surface!");
		}
	}

	void pickPhysicalDevice()
	{
		// choice the physical GPU device
		uint32_t deviceCount = 0;
		vkEnumeratePhysicalDevices(instance, &deviceCount, nullptr);
		if (deviceCount == 0)
		{
			throw std::runtime_error("failed to find GPUs with Vulkan support!");
		}

		std::vector<VkPhysicalDevice> devices(deviceCount);
		vkEnumeratePhysicalDevices(instance, &deviceCount, devices.data());

		// check if device suit for operation we want to perform

		for (const auto& device : devices)
		{
			if (isDeviceSuitable(device))
			{
				physicalDevice = device;
				break;
			}
		}

		if (physicalDevice == VK_NULL_HANDLE)
		{
			throw std::runtime_error("failed to find a suitable GPU!");
		}

		
	}

	bool isDeviceSuitable(VkPhysicalDevice device) 
	{
		// base device suitablility checks
		VkPhysicalDeviceProperties deviceProperties{};
		vkGetPhysicalDeviceProperties(device, &deviceProperties);

		std::cout << "device properties:\n"
			<< '\t' << "deviceName:\t" << deviceProperties.deviceName << '\n'
			<< '\t' << "deviceType:\t" << deviceProperties.deviceType << '\n'
			<< '\t' << "deviceID:\t" << deviceProperties.deviceID << '\n'
			<< '\t' << "vendorID:\t" << deviceProperties.vendorID << '\n'
			<< '\t' << "driverVersion:\t" << deviceProperties.driverVersion << '\n';


		VkPhysicalDeviceFeatures deviceFeatures{};
		vkGetPhysicalDeviceFeatures(device, &deviceFeatures);

		QueueFamilyIndices indices = findQueueFamilies(device);

		bool extensionsSupported = checkDeviceExtensionSupport(device);

		bool swapChainAdequate = false;
		if (extensionsSupported)
		{
			SwapChainSupportDetails swapChainSupport = querySwapChainSupport(device);
			swapChainAdequate = !swapChainSupport.formats.empty() && !swapChainSupport.presentModes.empty();
		}

		VkPhysicalDeviceFeatures supportedFeatures;
		vkGetPhysicalDeviceFeatures(device, &supportedFeatures);
		//return deviceProperties.deviceType == VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU && deviceFeatures.geometryShader;
		return indices.isComplete() && extensionsSupported && swapChainAdequate && supportedFeatures.samplerAnisotropy;
	}

	bool checkDeviceExtensionSupport(VkPhysicalDevice device)
	{
		uint32_t extensionCount;
		vkEnumerateDeviceExtensionProperties(device, nullptr, &extensionCount, nullptr);

		std::vector<VkExtensionProperties> availableExtensions(extensionCount);
		vkEnumerateDeviceExtensionProperties(device, nullptr, &extensionCount, availableExtensions.data());

		std::set<std::string> requiredExtensions(deviceExtensions.begin(), deviceExtensions.end());

		for (const auto& extension : availableExtensions) {
			requiredExtensions.erase(extension.extensionName);
		}

		return requiredExtensions.empty();
	}

	QueueFamilyIndices findQueueFamilies(VkPhysicalDevice device)
	{
		QueueFamilyIndices indices;

		uint32_t queueFamilyCount = 0;
		vkGetPhysicalDeviceQueueFamilyProperties(device, &queueFamilyCount, nullptr);

		std::vector<VkQueueFamilyProperties> queueFamilies(queueFamilyCount);
		vkGetPhysicalDeviceQueueFamilyProperties(device, &queueFamilyCount, queueFamilies.data());

		uint32_t idx = 0;
		for (const auto& queueFamily : queueFamilies)
		{
			if (queueFamily.queueFlags & VK_QUEUE_GRAPHICS_BIT)
			{
				indices.graphicsFamily = idx;
			}

			VkBool32 presentSupport = false;
			vkGetPhysicalDeviceSurfaceSupportKHR(device, idx, surface, &presentSupport);
			if (presentSupport)
			{
				indices.presentFamily = idx;
			}

			if (indices.isComplete())
			{
				break;
			}
			idx++;
		}

		return indices;
	}

	void createLogicalDevice()
	{
		QueueFamilyIndices indices = findQueueFamilies(physicalDevice);

		// specify the queues to create
		std::vector<VkDeviceQueueCreateInfo> queueCreateInfos{};
		std::set<uint32_t> uniqueQueueFamilies = { indices.graphicsFamily.value(), indices.presentFamily.value() };


		float queuePriority = 1.0f;  // -> in range [0, 1.0]
		for (uint32_t queueFamily : uniqueQueueFamilies)
		{
			VkDeviceQueueCreateInfo queueCreateInfo{};
			queueCreateInfo.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
			queueCreateInfo.queueFamilyIndex = queueFamily;
			queueCreateInfo.queueCount = 1;
			queueCreateInfo.pQueuePriorities = &queuePriority;

			queueCreateInfos.push_back(queueCreateInfo);
		}

		// specify used device features
		VkPhysicalDeviceFeatures deviceFeatures{};
		deviceFeatures.samplerAnisotropy = VK_TRUE;

		// create logical device
		VkDeviceCreateInfo createInfo{};
		createInfo.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;

		createInfo.queueCreateInfoCount = static_cast<uint32_t>(queueCreateInfos.size());
		createInfo.pQueueCreateInfos = queueCreateInfos.data();

		createInfo.pEnabledFeatures = &deviceFeatures;

		if (enbleValidationLayers)
		{
			createInfo.enabledLayerCount = static_cast<uint32_t> (validationLayers.size());
			createInfo.ppEnabledLayerNames = validationLayers.data();
		}
		else
		{
			createInfo.enabledLayerCount = 0;
		}

		// enable device extension
		createInfo.enabledExtensionCount = static_cast<uint32_t>(deviceExtensions.size());
		createInfo.ppEnabledExtensionNames = deviceExtensions.data();

		// create logical device
		if (vkCreateDevice(physicalDevice, &createInfo, nullptr, &device) != VK_SUCCESS)
		{
			throw std::runtime_error("failed to create logical device!");
		}

		// retrieving queue handles
		vkGetDeviceQueue(device, indices.graphicsFamily.value(), 0, &graphicsQueue);
		vkGetDeviceQueue(device, indices.presentFamily.value(), 0, &presentQueue);
	}

	SwapChainSupportDetails querySwapChainSupport(VkPhysicalDevice device)
	{
		// query the swapChain supported details
		SwapChainSupportDetails details{};

		vkGetPhysicalDeviceSurfaceCapabilitiesKHR(device, surface, &details.capabilities);

		uint32_t formatCount = 0;
		vkGetPhysicalDeviceSurfaceFormatsKHR(device, surface, &formatCount, nullptr);
		if (formatCount != 0)
		{
			details.formats.resize(formatCount);
			vkGetPhysicalDeviceSurfaceFormatsKHR(device, surface, &formatCount, details.formats.data());
		}

		uint32_t presentModeCount = 0;
		vkGetPhysicalDeviceSurfacePresentModesKHR(device, surface, &presentModeCount, nullptr);
		if (presentModeCount != 0)
		{
			details.presentModes.resize(presentModeCount);
			vkGetPhysicalDeviceSurfacePresentModesKHR(device, surface, &presentModeCount, details.presentModes.data());
		}

		return details;
	}

	VkSurfaceFormatKHR chooseSwapSurfaceFormat(const std::vector<VkSurfaceFormatKHR>& availableFormats)
	{
		// surface image format, color sapce and format
		for (const auto& availableFormat : availableFormats)
		{
			if (availableFormat.format == VK_FORMAT_B8G8R8A8_SRGB && availableFormat.colorSpace == VK_COLOR_SPACE_SRGB_NONLINEAR_KHR)
			{
				return availableFormat;
			}
		}
		return availableFormats[0];
	}

	VkPresentModeKHR chooseSwapPresentMode(const std::vector<VkPresentModeKHR>& availablePrensentModes)
	{
		// how to swap between different frame
		for (const auto& presentMode : availablePrensentModes)
		{
			if (presentMode == VK_PRESENT_MODE_MAILBOX_KHR)
			{
				return presentMode;
			}
		}
		return VK_PRESENT_MODE_FIFO_KHR;
	}

	VkExtent2D chooseSwapExtent(const VkSurfaceCapabilitiesKHR& capabilities)
	{
		// the resolution of the chain images, same as the window's resolution is pixel
		if (capabilities.currentExtent.width != (std::numeric_limits<uint32_t>::max)())
		{
			return capabilities.currentExtent;
		}
		
		// get actual window size
		int width, height;
		glfwGetFramebufferSize(window, &width, &height);

		VkExtent2D actualExtent = {
			static_cast<uint32_t>(width),
			static_cast<uint32_t>(height)
		};

		actualExtent.width = std::clamp(actualExtent.width, capabilities.minImageExtent.width, capabilities.maxImageExtent.width);
		actualExtent.height = std::clamp(actualExtent.height, capabilities.minImageExtent.height, capabilities.maxImageExtent.height);

		return actualExtent;
	}

	void creataSwapChain()
	{
		SwapChainSupportDetails swapChainSupport = querySwapChainSupport(physicalDevice);

		VkSurfaceFormatKHR surfaceFormat = chooseSwapSurfaceFormat(swapChainSupport.formats);
		VkPresentModeKHR presentMode = chooseSwapPresentMode(swapChainSupport.presentModes);
		VkExtent2D extent = chooseSwapExtent(swapChainSupport.capabilities);

		uint32_t imageCount = swapChainSupport.capabilities.minImageCount + 1; // at least double buffer

		// make sure imageCount not large than maxImageCount
		imageCount = std::min(imageCount, swapChainSupport.capabilities.maxImageCount);

		VkSwapchainCreateInfoKHR createInfo{};
		createInfo.sType = VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR;
		createInfo.surface = surface;

		createInfo.minImageCount = imageCount;
		createInfo.imageFormat = surfaceFormat.format;
		createInfo.imageColorSpace = surfaceFormat.colorSpace;
		createInfo.imageExtent = extent;
		createInfo.imageArrayLayers = 1;
		createInfo.imageUsage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT;

		// if graphics queue and present queue are different
		// need specify how to handle swap chain images
		QueueFamilyIndices indices = findQueueFamilies(physicalDevice);
		uint32_t queueFamilyIndices[] = { indices.graphicsFamily.value(), indices.presentFamily.value() };

		if (indices.graphicsFamily != indices.presentFamily)
		{
			createInfo.imageSharingMode = VK_SHARING_MODE_CONCURRENT;
			createInfo.queueFamilyIndexCount = 2;
			createInfo.pQueueFamilyIndices = queueFamilyIndices;
		}
		else
		{
			createInfo.imageSharingMode = VK_SHARING_MODE_EXCLUSIVE;
			createInfo.queueFamilyIndexCount = 0;
			createInfo.pQueueFamilyIndices = nullptr;
		}

		// specify transform apply to image in swap chain
		createInfo.preTransform = swapChainSupport.capabilities.currentTransform;

		createInfo.compositeAlpha = VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR;

		createInfo.presentMode = presentMode;
		createInfo.clipped = VK_TRUE;
		createInfo.oldSwapchain = VK_NULL_HANDLE;
		
		// create swap chain
		if (vkCreateSwapchainKHR(device, &createInfo, nullptr, &swapChain) != VK_SUCCESS)
		{
			throw std::runtime_error("failed to create swap chain!");
		}

		// retrieving the swap chain images
		vkGetSwapchainImagesKHR(device, swapChain, &imageCount, nullptr);
		swapChainImages.resize(imageCount);
		vkGetSwapchainImagesKHR(device, swapChain, &imageCount, swapChainImages.data());

		// save some information
		swapChainImageFormat = surfaceFormat.format;
		swapChainExtent = extent;
	}

	void createSwapChainImageViews()
	{
		swapChainImageViews.resize(swapChainImages.size());

		for (size_t idx = 0; idx < swapChainImageViews.size(); idx++)
		{
			swapChainImageViews[idx] = createImageView(swapChainImages[idx], swapChainImageFormat, VK_IMAGE_ASPECT_COLOR_BIT);
		}
	}

	void createRenderPass()
	{
		// color attachment
		VkAttachmentDescription colorAttachment {
			.format = swapChainImageFormat,
			.samples = VK_SAMPLE_COUNT_1_BIT,

			.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR,
			.storeOp = VK_ATTACHMENT_STORE_OP_STORE,

			.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE,
			.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE,

			.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED,
			.finalLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR,
		};


		// attachment references
		VkAttachmentReference colorAttachmentRef {
			.attachment = 0,
			.layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL
		};

		// depth attachment
		VkAttachmentDescription depthAttachment {
			.format = findDepthFormat(),
			.samples = VK_SAMPLE_COUNT_1_BIT,
			.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR,
			.storeOp = VK_ATTACHMENT_STORE_OP_DONT_CARE,
			.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE,
			.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE,
			.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED,
			.finalLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
		};

		VkAttachmentReference depthAttachmentRef{
			.attachment = 1,
			.layout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL
		};

		// subpass
		VkSubpassDescription subpass {
			.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS,
			.colorAttachmentCount = 1,
			.pColorAttachments = &colorAttachmentRef,
			.pDepthStencilAttachment = &depthAttachmentRef
		};

		// subpass dependencies
		VkSubpassDependency dependency {
			.srcSubpass = VK_SUBPASS_EXTERNAL,
			.dstSubpass = 0,
			.srcStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT | VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT,
			.dstStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT | VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT,
			.srcAccessMask = 0,
			.dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT | VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT,
		};

		std::array<VkAttachmentDescription, 2> attachments = { colorAttachment, depthAttachment };

		VkRenderPassCreateInfo renderPassInfo {
			.sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO,
			.attachmentCount = static_cast<uint32_t>(attachments.size()),
			.pAttachments = attachments.data(),
			.subpassCount = 1,
			.pSubpasses = &subpass,
			.dependencyCount = 1,
			.pDependencies = &dependency,
		};

		if (vkCreateRenderPass(device, &renderPassInfo, nullptr, &renderPass) != VK_SUCCESS) {
			throw std::runtime_error("failed to create render pass!");
		}
	}

	VkDescriptorSetLayout createDescriptorSetLayout(uint32_t samplerNum)
	{
		std::vector<VkDescriptorSetLayoutBinding> bindings; // { uboLayoutBinding, samplerLayoutBinding };
		VkDescriptorSetLayoutBinding uboLayoutBinding {
			.binding            = 0,
			.descriptorType     = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
			.descriptorCount    = 1,
			.stageFlags         = VK_SHADER_STAGE_VERTEX_BIT, // ubo in vertex shader
			.pImmutableSamplers = nullptr, // Optional
		};
		bindings.push_back(uboLayoutBinding);

		// texture 
		for (size_t idx = 0; idx < samplerNum; idx++)
		{
			VkDescriptorSetLayoutBinding samplerLayoutBinding{
				.binding            = static_cast<uint32_t>(1 + idx),
				.descriptorType     = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
				.descriptorCount    = 1,
				.stageFlags         = VK_SHADER_STAGE_FRAGMENT_BIT,
				.pImmutableSamplers = nullptr,
			};
			bindings.push_back(samplerLayoutBinding);
		}

		// shadowMap texture
		VkDescriptorSetLayoutBinding shadowMapBinding{
				.binding            = 2,
				.descriptorType     = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
				.descriptorCount    = 1,
				.stageFlags         = VK_SHADER_STAGE_FRAGMENT_BIT,
				.pImmutableSamplers = nullptr,
		};
		bindings.push_back(shadowMapBinding);

		VkDescriptorSetLayoutBinding viewUboLayoutBinding{
			.binding            = 3,
			.descriptorType     = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
			.descriptorCount    = 1,
			.stageFlags         = VK_SHADER_STAGE_FRAGMENT_BIT, // ubo in vertex shader
			.pImmutableSamplers = nullptr, // Optional
		};
		bindings.push_back(viewUboLayoutBinding);

		VkDescriptorSetLayoutCreateInfo layoutInfo {
			.sType        = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
			.bindingCount = static_cast<uint32_t>(bindings.size()),
			.pBindings    = bindings.data(),
		};

		VkDescriptorSetLayout layout;
		if (vkCreateDescriptorSetLayout(device, &layoutInfo, nullptr, &layout) != VK_SUCCESS) {
			throw std::runtime_error("failed to create descriptor set layout!");
		}
		return layout;
	}

	VkPipeline createGraphicsPipeline(const VkPipelineLayout& pipelineLayout)
	{
		// create shader module
		std::vector<char> vertShaderCode = readFile(vertShaderPath);
		std::vector<char> fragShaderCode = readFile(fragShaderPath);

		VkShaderModule vertShaderModule = createShaderModule(vertShaderCode);
		VkShaderModule fragShaderModule = createShaderModule(fragShaderCode);


		// shader stage creation
		VkPipelineShaderStageCreateInfo vertShaderStageInfo {
			.sType  = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
			.stage  = VK_SHADER_STAGE_VERTEX_BIT,
			.module = vertShaderModule,
			.pName  = "main",
		};

		VkPipelineShaderStageCreateInfo fragShaderStageInfo {
			.sType  = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
			.stage  = VK_SHADER_STAGE_FRAGMENT_BIT,
			.module = fragShaderModule,
			.pName  = "main",
		};

		
		VkPipelineShaderStageCreateInfo shaderStages[] = { vertShaderStageInfo, fragShaderStageInfo };

		// dynamic state
		std::vector<VkDynamicState> dynamicSates = {
			VK_DYNAMIC_STATE_VIEWPORT,
			VK_DYNAMIC_STATE_SCISSOR
		};
		VkPipelineDynamicStateCreateInfo dynamicState {
			.sType             = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO,
			.dynamicStateCount = static_cast<uint32_t>(dynamicSates.size()),
			.pDynamicStates    = dynamicSates.data()
		};


		// vertex input
		auto bindingDescription = Vertex::getBindingDescription();
		auto attributeDescriptions = Vertex::getAttributeDescription();

		VkPipelineVertexInputStateCreateInfo vertexInputInfo {
			.sType                           = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO,
			.vertexBindingDescriptionCount   = 1,
			.pVertexBindingDescriptions      = &bindingDescription, // Optional

			.vertexAttributeDescriptionCount = static_cast<uint32_t>(attributeDescriptions.size()),
			.pVertexAttributeDescriptions    = attributeDescriptions.data(), // Optional
		};


		// input assembly
		VkPipelineInputAssemblyStateCreateInfo inputAssembly {
			.sType                  = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO,
			.topology               = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST,
			.primitiveRestartEnable = VK_FALSE,
		};


		// viewport and scissors, image to framebuffer
		VkViewport viewport {
			.x        = 0.0f,
			.y        = 0.0f,
			.width    = (float)swapChainExtent.width,
			.height   = (float)swapChainExtent.height,
			.minDepth = 0.0f, // [0.0f, 1.0f]
			.maxDepth = 1.0f, // [0.0f, 1.0f]
		}; 

		// which regions pixels will actually be stored. 
		VkRect2D scissor {
			.offset = { 0, 0 },
			.extent = swapChainExtent,
		};

		VkPipelineViewportStateCreateInfo viewportState {
			.sType         = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO,
			.viewportCount = 1,
			.pViewports    = &viewport,
			.scissorCount  = 1,
			.pScissors     = &scissor,
		};


		// rasterization
		VkPipelineRasterizationStateCreateInfo rasterizer {
			.sType                   = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO,
			.depthClampEnable        = VK_FALSE,
			.rasterizerDiscardEnable = VK_FALSE, // fragment to framebuffer
			.polygonMode             = VK_POLYGON_MODE_FILL,
			.cullMode                = VK_CULL_MODE_BACK_BIT,
			.frontFace               = VK_FRONT_FACE_COUNTER_CLOCKWISE,
			.depthBiasEnable         = VK_FALSE,
			.depthBiasConstantFactor = 0.0f, // Optional
			.depthBiasClamp          = 0.0f, // Optional
			.depthBiasSlopeFactor    = 0.0f, // Optional
			.lineWidth               = 1.0f,
		};

		VkPipelineMultisampleStateCreateInfo multisampling {
			.sType                 = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO,
			.rasterizationSamples  = VK_SAMPLE_COUNT_1_BIT,
			.sampleShadingEnable   = VK_FALSE,
			.minSampleShading      = 1.0f, // Optional
			.pSampleMask           = nullptr, // Optional
			.alphaToCoverageEnable = VK_FALSE, // Optional
			.alphaToOneEnable      = VK_FALSE, // Optional
		};

		// color blending
		VkPipelineColorBlendAttachmentState colorBlendAttachment {
			.blendEnable         = VK_FALSE,
			.srcColorBlendFactor = VK_BLEND_FACTOR_ONE, // Optional
			.dstColorBlendFactor = VK_BLEND_FACTOR_ZERO, // Optional
			.colorBlendOp        = VK_BLEND_OP_ADD, // Optional
			.srcAlphaBlendFactor = VK_BLEND_FACTOR_ONE, // Optional
			.dstAlphaBlendFactor = VK_BLEND_FACTOR_ZERO, // Optional
			.alphaBlendOp        = VK_BLEND_OP_ADD, // Optional
			.colorWriteMask      = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT | VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT,
		};

		VkPipelineColorBlendStateCreateInfo colorBlending {
			.sType             = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO,
			.logicOpEnable     = VK_FALSE,
			.logicOp           = VK_LOGIC_OP_COPY, // Optional
			.attachmentCount   = 1,
			.pAttachments      = &colorBlendAttachment,
			.blendConstants    = { 0.0f }, // Optional
		};

		// depth and stencil
		VkPipelineDepthStencilStateCreateInfo depthStencil{
			.sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO,
			.depthTestEnable = VK_TRUE,
			.depthWriteEnable = VK_TRUE,
			.depthCompareOp = VK_COMPARE_OP_LESS,
			.depthBoundsTestEnable = VK_FALSE,
			.stencilTestEnable = VK_FALSE,
			.front = {},
			.back = {},
			.minDepthBounds = 0.0f, // Optional
			.maxDepthBounds = 1.0f, // Optional
		};



		// create pipeline
		VkGraphicsPipelineCreateInfo pipelineInfo {
			.sType               = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO,
			.stageCount          = 2,
			.pStages             = shaderStages,

			.pVertexInputState   = &vertexInputInfo,
			.pInputAssemblyState = &inputAssembly,
			.pViewportState      = &viewportState,
			.pRasterizationState = &rasterizer,
			.pMultisampleState   = &multisampling,
			.pDepthStencilState  = &depthStencil, // Optional
			.pColorBlendState    = &colorBlending,
			.pDynamicState       = &dynamicState,

			.layout              = pipelineLayout,

			.renderPass          = renderPass,
			.subpass             = 0,

			.basePipelineHandle  = VK_NULL_HANDLE, // Optional
			.basePipelineIndex   = -1, // Optional
		};

		VkPipeline pipeline;
		if (vkCreateGraphicsPipelines(device, VK_NULL_HANDLE, 1, &pipelineInfo, nullptr, &pipeline) != VK_SUCCESS) {
			throw std::runtime_error("failed to create graphics pipeline!");
		}

		// clean up resources
		vkDestroyShaderModule(device, fragShaderModule, nullptr);
		vkDestroyShaderModule(device, vertShaderModule, nullptr);

		return pipeline;
	}

	VkShaderModule createShaderModule(const std::vector<char>& code)
	{
		VkShaderModuleCreateInfo createInfo{};
		createInfo.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
		createInfo.codeSize = code.size();
		createInfo.pCode = reinterpret_cast<const uint32_t*>(code.data());

		VkShaderModule shaderModule;
		if (vkCreateShaderModule(device, &createInfo, nullptr, &shaderModule) != VK_SUCCESS)
		{
			throw std::runtime_error("failed to create shader module");
		}
		return shaderModule;
	}


	void createFramebuffers()
	{
		swapChainFramebuffers.resize(swapChainImageViews.size());

		for (size_t idx = 0; idx < swapChainImageViews.size(); idx++)
		{
			std::array<VkImageView, 2> attachments = { 
				swapChainImageViews[idx],
				depthImageView
			};

			VkFramebufferCreateInfo framebufferInfo {
				.sType           = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO,
				.renderPass      = renderPass,
				.attachmentCount = static_cast<uint32_t>(attachments.size()),
				.pAttachments    = attachments.data(),
				.width           = swapChainExtent.width,
				.height          = swapChainExtent.height,
				.layers          = 1,
			};

			if (vkCreateFramebuffer(device, &framebufferInfo, nullptr, &swapChainFramebuffers[idx]) != VK_SUCCESS)
			{
				throw std::runtime_error("failed to create framebuffer!");
			}
		}
	}

	void createCommandPool()
	{
		QueueFamilyIndices queueFamilyIndices = findQueueFamilies(physicalDevice);

		VkCommandPoolCreateInfo poolInfo{};
		poolInfo.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
		poolInfo.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
		poolInfo.queueFamilyIndex = queueFamilyIndices.graphicsFamily.value();

		if (vkCreateCommandPool(device, &poolInfo, nullptr, &commandPool) != VK_SUCCESS)
		{
			throw std::runtime_error("failed to create command pool!");
		}
	}


	uint32_t findMemoryType(uint32_t typeFilter, VkMemoryPropertyFlags properties)
	{
		// find physical device memory support 
		VkPhysicalDeviceMemoryProperties memProperties;
		vkGetPhysicalDeviceMemoryProperties(physicalDevice, &memProperties);

		for (uint32_t idx = 0; idx < memProperties.memoryTypeCount; idx++)
		{
			if (typeFilter & (1 << idx) && (memProperties.memoryTypes[idx].propertyFlags & properties) == properties)
			{
				return idx;
			}
		}
		throw std::runtime_error("failed to find suitable memory type!");
	}

	void createBuffer(VkDeviceSize size, VkBufferUsageFlags usage, VkMemoryPropertyFlags properties, VkBuffer& buffer, VkDeviceMemory& bufferMemory)
	{
		VkBufferCreateInfo bufferInfo{};
		bufferInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
		bufferInfo.size = size;

		bufferInfo.usage = usage;
		bufferInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

		if (vkCreateBuffer(device, &bufferInfo, nullptr, &buffer) != VK_SUCCESS)
		{
			throw std::runtime_error("failed to create vertex buffer!");
		}

		// memory requirement
		VkMemoryRequirements memRequirements;
		vkGetBufferMemoryRequirements(device, buffer, &memRequirements);

		// allocation
		VkMemoryAllocateInfo allocInfo{};
		allocInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
		allocInfo.allocationSize = memRequirements.size;
		allocInfo.memoryTypeIndex = findMemoryType(memRequirements.memoryTypeBits, properties);

		if (vkAllocateMemory(device, &allocInfo, nullptr, &bufferMemory) != VK_SUCCESS)
		{
			throw std::runtime_error("failed to allocate vertex buffer memory!");
		}

		// binding memory with buffer
		vkBindBufferMemory(device, buffer, bufferMemory, 0);
	}


	void createImage(uint32_t width, uint32_t height, VkFormat format, VkImageTiling tiling, VkImageUsageFlags usage, 
		VkMemoryPropertyFlags properties, VkImage& image, VkDeviceMemory& imageMemory)
	{
		VkImageCreateInfo imageInfo {
			.sType         = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO,
			.imageType     = VK_IMAGE_TYPE_2D,
			.format        = format,
			.extent        =  {
				.width     = width,
				.height    = height,
				.depth     = 1,
			},
			.mipLevels     = 1,
			.arrayLayers   = 1,
			.samples       = VK_SAMPLE_COUNT_1_BIT,
			.tiling        = tiling,
			.usage         = usage,
			.sharingMode   = VK_SHARING_MODE_EXCLUSIVE,
			.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED,
		};

		if (vkCreateImage(device, &imageInfo, nullptr, &image) != VK_SUCCESS) {
			throw std::runtime_error("failed to create image!");
		}

		VkMemoryRequirements memRequirements;
		vkGetImageMemoryRequirements(device, image, &memRequirements);

		VkMemoryAllocateInfo allocInfo {
			.sType           = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
			.allocationSize  = memRequirements.size,
			.memoryTypeIndex = findMemoryType(memRequirements.memoryTypeBits, properties),
		};

		if (vkAllocateMemory(device, &allocInfo, nullptr, &imageMemory) != VK_SUCCESS) {
			throw std::runtime_error("failed to allocate image memory!");
		}

		vkBindImageMemory(device, image, imageMemory, 0);
	}

	VkCommandBuffer beginSingleTimeCommands() {
		VkCommandBufferAllocateInfo allocInfo {
			.sType              = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
			.commandPool        = commandPool,
			.level              = VK_COMMAND_BUFFER_LEVEL_PRIMARY,
			.commandBufferCount = 1,
		};

		VkCommandBuffer commandBuffer;
		vkAllocateCommandBuffers(device, &allocInfo, &commandBuffer);

		VkCommandBufferBeginInfo beginInfo {
			.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
			.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT,
		};

		vkBeginCommandBuffer(commandBuffer, &beginInfo);

		return commandBuffer;
	}

	void endSingleTimeCommands(VkCommandBuffer commandBuffer) {
		vkEndCommandBuffer(commandBuffer);

		VkSubmitInfo submitInfo {
			.sType              = VK_STRUCTURE_TYPE_SUBMIT_INFO,
			.commandBufferCount = 1,
			.pCommandBuffers    = &commandBuffer,
		};

		vkQueueSubmit(graphicsQueue, 1, &submitInfo, VK_NULL_HANDLE);
		vkQueueWaitIdle(graphicsQueue);

		vkFreeCommandBuffers(device, commandPool, 1, &commandBuffer);
	}


	VkFormat findSupportedFormat(const std::vector<VkFormat>& candidates, VkImageTiling tiling, VkFormatFeatureFlags features)
	{
		for (VkFormat format : candidates) {
			VkFormatProperties props;
			vkGetPhysicalDeviceFormatProperties(physicalDevice, format, &props);

			if (tiling == VK_IMAGE_TILING_LINEAR && (props.linearTilingFeatures & features) == features) 
			{
				return format;
			}
			else if (tiling == VK_IMAGE_TILING_OPTIMAL && (props.optimalTilingFeatures & features) == features) 
			{
				return format;
			}
		}
		throw std::runtime_error("failed to find supported format!");
	}

	VkFormat findDepthFormat()
	{
		return findSupportedFormat(
			{ VK_FORMAT_D32_SFLOAT, VK_FORMAT_D32_SFLOAT_S8_UINT, VK_FORMAT_D24_UNORM_S8_UINT },
			VK_IMAGE_TILING_OPTIMAL,
			VK_FORMAT_FEATURE_DEPTH_STENCIL_ATTACHMENT_BIT
		);
	}

	bool hasStencilComponent(VkFormat format)
	{
		return format == VK_FORMAT_D32_SFLOAT_S8_UINT || format == VK_FORMAT_D24_UNORM_S8_UINT;
	}

	void createDepthResources()
	{
		VkFormat depthFormat = findDepthFormat();

		createImage(swapChainExtent.width, swapChainExtent.height, depthFormat,
			VK_IMAGE_TILING_OPTIMAL, VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
			depthImage, depthImageMemory);

		depthImageView = createImageView(depthImage, depthFormat, VK_IMAGE_ASPECT_DEPTH_BIT);

		// bug
		// transitionImageLayout(depthImage, depthFormat, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL);
	}

	void createTextureImage(const std::string& texImgPath, VkImage& texImage, VkDeviceMemory& texImageMemory)
	{
		int texWidth, texHeight, texChannels;
		stbi_uc* pixels = stbi_load(texImgPath.c_str(), &texWidth, &texHeight, &texChannels, STBI_rgb_alpha);

		VkDeviceSize imageSize = texWidth * texHeight * 4;
		if (!pixels)
		{
			throw std::runtime_error("failed to load texture image!");
		}

		VkBuffer stagingBuffer{};
		VkDeviceMemory stagingBufferMemory{};

		// texture staging buffer
		createBuffer(imageSize, VK_BUFFER_USAGE_TRANSFER_SRC_BIT, 
			VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, stagingBuffer, stagingBufferMemory);

		void* data;
		vkMapMemory(device, stagingBufferMemory, 0, imageSize, 0, &data);
			memcpy(data, pixels, static_cast<size_t>(imageSize));
		vkUnmapMemory(device, stagingBufferMemory);

		stbi_image_free(pixels);

		createImage(texWidth, texHeight, VK_FORMAT_R8G8B8A8_SRGB, VK_IMAGE_TILING_OPTIMAL, 
			VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, texImage, texImageMemory);

		transitionImageLayout(texImage, VK_FORMAT_R8G8B8A8_SRGB, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL);
		copyBufferToImage(stagingBuffer, texImage, static_cast<uint32_t>(texWidth), static_cast<uint32_t>(texHeight));

		transitionImageLayout(texImage, VK_FORMAT_R8G8B8A8_SRGB, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);

		vkDestroyBuffer(device, stagingBuffer, nullptr);
		vkFreeMemory(device, stagingBufferMemory, nullptr);
	}

	void transitionImageLayout(VkImage image, VkFormat format, VkImageLayout oldLayout, VkImageLayout newLayout)
	{
		VkCommandBuffer commandBuffer = beginSingleTimeCommands();

		VkImageMemoryBarrier barrier {
			.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
			.oldLayout = oldLayout,
			.newLayout = newLayout,

			.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
			.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,

			.image = image,
			.subresourceRange = {
				.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
				.baseMipLevel = 0,
				.levelCount = 1,
				.baseArrayLayer = 0,
				.layerCount = 1
			}
		};

		VkPipelineStageFlags sourceStage;
		VkPipelineStageFlags destinationStage;

		if (newLayout == VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL)
		{
			barrier.subresourceRange.aspectMask = VK_IMAGE_ASPECT_DEPTH_BIT;

			if (hasStencilComponent(format))
			{
				barrier.subresourceRange.aspectMask |= VK_IMAGE_ASPECT_STENCIL_BIT;
			}
		}
		else
		{
			barrier.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
		}

		if (oldLayout == VK_IMAGE_LAYOUT_UNDEFINED && newLayout == VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL) 
		{
			barrier.srcAccessMask = 0;
			barrier.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;

			sourceStage = VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT;
			destinationStage = VK_PIPELINE_STAGE_TRANSFER_BIT;
		}
		else if (oldLayout == VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL && newLayout == VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL) 
		{
			barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
			barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;

			sourceStage = VK_PIPELINE_STAGE_TRANSFER_BIT;
			destinationStage = VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT;
		}
		else if (oldLayout == VK_IMAGE_LAYOUT_UNDEFINED && newLayout == VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL) 
		{
			barrier.srcAccessMask = 0;
			barrier.dstAccessMask = VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_READ_BIT | VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;

			sourceStage = VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT;
			destinationStage = VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT;
		}
		else 
		{
			throw std::invalid_argument("unsupported layout transition!");
		}

		

		vkCmdPipelineBarrier(
			commandBuffer,
			sourceStage, destinationStage,
			0,
			0, nullptr,
			0, nullptr,
			1, &barrier
		);

		endSingleTimeCommands(commandBuffer);
	}

	void copyBufferToImage(VkBuffer buffer, VkImage image, uint32_t width, uint32_t height) {
		VkCommandBuffer commandBuffer = beginSingleTimeCommands();

		VkBufferImageCopy region{};
		region.bufferOffset = 0;
		region.bufferRowLength = 0;
		region.bufferImageHeight = 0;

		region.imageSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
		region.imageSubresource.mipLevel = 0;
		region.imageSubresource.baseArrayLayer = 0;
		region.imageSubresource.layerCount = 1;

		region.imageOffset = { 0, 0, 0 };
		region.imageExtent = {
			width,
			height,
			1
		};

		vkCmdCopyBufferToImage(
			commandBuffer,
			buffer,
			image,
			VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
			1,
			&region
		);

		endSingleTimeCommands(commandBuffer);
	}

	VkImageView createImageView(VkImage image, VkFormat format, VkImageAspectFlags aspectFlags)
	{
		VkImageViewCreateInfo viewInfo {
			.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,
			.image = image,
			.viewType = VK_IMAGE_VIEW_TYPE_2D,
			.format = format,

			.subresourceRange = {
				.aspectMask = aspectFlags,
				.baseMipLevel = 0,
				.levelCount = 1,
				.baseArrayLayer = 0,
				.layerCount = 1,
			}
		};


		VkImageView imageView;
		if (vkCreateImageView(device, &viewInfo, nullptr, &imageView) != VK_SUCCESS) {
			throw std::runtime_error("failed to create texture image view!");
		}

		return imageView;
	}

	VkImageView createTextureImageView(VkImage image)
	{
		return createImageView(image, VK_FORMAT_R8G8B8A8_SRGB, VK_IMAGE_ASPECT_COLOR_BIT);
	}

	VkSampler createTextureSampler(
		VkFilter filter = VK_FILTER_LINEAR,
		VkSamplerAddressMode  addressModeU = VK_SAMPLER_ADDRESS_MODE_REPEAT,
		VkSamplerAddressMode  addressModeV = VK_SAMPLER_ADDRESS_MODE_REPEAT,
		VkSamplerAddressMode  addressModeW = VK_SAMPLER_ADDRESS_MODE_REPEAT,
		VkBorderColor borderColor = VK_BORDER_COLOR_INT_OPAQUE_BLACK,
		uint32_t mipLevels = 1
	)
	{
		VkSampler texSampler;

		VkPhysicalDeviceProperties properties{};
		vkGetPhysicalDeviceProperties(physicalDevice, &properties);

		VkSamplerCreateInfo samplerInfo {
			.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO,
			.magFilter = filter,
			.minFilter = filter,

			.mipmapMode = VK_SAMPLER_MIPMAP_MODE_LINEAR,

			.addressModeU = addressModeU,
			.addressModeV = addressModeV,
			.addressModeW = addressModeW,

			.mipLodBias = 0.0f,

			.anisotropyEnable = VK_TRUE,
			.maxAnisotropy = properties.limits.maxSamplerAnisotropy,

			.compareEnable = VK_FALSE,
			.compareOp = VK_COMPARE_OP_ALWAYS,
			
			.minLod = 0.0f,
			.maxLod = static_cast<float>(mipLevels),
			
			.borderColor = borderColor,
			.unnormalizedCoordinates = VK_FALSE,
		};

		if (vkCreateSampler(device, &samplerInfo, nullptr, &texSampler) != VK_SUCCESS)
		{
			throw std::runtime_error("failed to create texture sampler!");
		}

		return texSampler;
	}

	void loadModel(const std::string&path, RenderObject& model)
	{
		tinyobj::attrib_t attrib;
		std::vector<tinyobj::shape_t> shapes;
		std::vector<tinyobj::material_t> materials;
		std::string warn, err;

		if (!tinyobj::LoadObj(&attrib, &shapes, &materials, &warn, &err, path.c_str())) {
			throw std::runtime_error(warn + err);
		}

		std::unordered_map<Vertex, uint32_t> uniqueVertices{};

		for (const auto& shape : shapes) {
			for (const auto& index : shape.mesh.indices) {
				Vertex vertex{};

				vertex.pos = {
					attrib.vertices[3 * index.vertex_index + 0],
					attrib.vertices[3 * index.vertex_index + 1],
					attrib.vertices[3 * index.vertex_index + 2]
				};

				vertex.texCoord = {
					attrib.texcoords[2 * index.texcoord_index + 0],
					1.0f - attrib.texcoords[2 * index.texcoord_index + 1]
				};

				vertex.normal = {
					attrib.normals[3 * index.normal_index + 0],
					attrib.normals[3 * index.normal_index + 1],
					attrib.normals[3 * index.normal_index + 2]
				};

				vertex.color = { 1.0f, 1.0f, 1.0f };

				// Flip Y-Axis of vertex positions
				//vertex.pos.y *= -1.0f;
				//vertex.normal.y *= -1.0f;

				if (uniqueVertices.count(vertex) == 0)
				{
					uniqueVertices[vertex] = static_cast<uint32_t>(model.vertices.size());
					model.vertices.push_back(vertex);
				}
				model.indices.push_back(uniqueVertices[vertex]);
			}
		}
	}

	void createVertexBuffer(const std::vector<Vertex>& vertices, VkBuffer & vBuffer, VkDeviceMemory& vBufferMemory)
	{
		// create buffer
		VkDeviceSize bufferSize = sizeof(vertices[0]) * vertices.size();

		// using staging buffer
		VkBuffer stagingBuffer{};
		VkDeviceMemory stagingBufferMemory{};
		createBuffer(bufferSize, VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
			VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, stagingBuffer, stagingBufferMemory);

		// fill data to vertex buffer
		void* data;
		vkMapMemory(device, stagingBufferMemory, 0, bufferSize, 0, &data);
			memcpy(data, vertices.data(), (size_t)bufferSize);
		vkUnmapMemory(device, stagingBufferMemory);

		// destination 

		createBuffer(bufferSize, 
			VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_VERTEX_BUFFER_BIT, 
			VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, vBuffer, vBufferMemory);

		// copy stagingBuffer to vertexBuffer
		copyBuffer(stagingBuffer, vBuffer, bufferSize);

		vkDestroyBuffer(device, stagingBuffer, nullptr);
		vkFreeMemory(device, stagingBufferMemory, nullptr);
	}

	void createIndexBuffer(const std::vector<uint32_t>& indices, VkBuffer& iBuffer, VkDeviceMemory& iBufferMemory)
	{
		VkDeviceSize bufferSize = sizeof(indices[0]) * indices.size();

		VkBuffer stagingBuffer;
		VkDeviceMemory stagingBufferMemory;
		createBuffer(bufferSize, VK_BUFFER_USAGE_TRANSFER_SRC_BIT, 
			VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, stagingBuffer, stagingBufferMemory);

		void* data;
		vkMapMemory(device, stagingBufferMemory, 0, bufferSize, 0, &data);
			memcpy(data, indices.data(), (size_t)bufferSize);
		vkUnmapMemory(device, stagingBufferMemory);

		createBuffer(bufferSize, 
			VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_INDEX_BUFFER_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, iBuffer, iBufferMemory);

		copyBuffer(stagingBuffer, iBuffer, bufferSize);

		vkDestroyBuffer(device, stagingBuffer, nullptr);
		vkFreeMemory(device, stagingBufferMemory, nullptr);
	}

	void copyBuffer(VkBuffer srcBuffer, VkBuffer dstBuffer, VkDeviceSize size)
	{
		VkCommandBuffer commandBuffer = beginSingleTimeCommands();

		VkBufferCopy copyRegion{};
		copyRegion.srcOffset = 0; // Optional
		copyRegion.dstOffset = 0; // Optional
		copyRegion.size = size;
		vkCmdCopyBuffer(commandBuffer, srcBuffer, dstBuffer, 1, &copyRegion);

		endSingleTimeCommands(commandBuffer);
	}

	void createUniformBuffers()
	{
		VkDeviceSize bufferSize = sizeof(UniformBufferObject);

		m_uniformBuffers.resize(MAX_FRAMES_IN_FLIGHT);
		m_uniformBuffersMemory.resize(MAX_FRAMES_IN_FLIGHT);
		m_uniformBuffersMapped.resize(MAX_FRAMES_IN_FLIGHT);

		for (size_t idx = 0; idx < MAX_FRAMES_IN_FLIGHT; idx++)
		{
			createBuffer(bufferSize, VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT, 
				VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, m_uniformBuffers[idx], m_uniformBuffersMemory[idx]);

			vkMapMemory(device, m_uniformBuffersMemory[idx], 0, bufferSize, 0, &m_uniformBuffersMapped[idx]);
		}

		bufferSize = sizeof(ViewUniformBufferObject);
		m_viewUniformBuffers.resize(MAX_FRAMES_IN_FLIGHT);
		m_viewUniformBuffersMemory.resize(MAX_FRAMES_IN_FLIGHT);
		m_viewUniformBuffersMapped.resize(MAX_FRAMES_IN_FLIGHT);

		for (size_t idx = 0; idx < MAX_FRAMES_IN_FLIGHT; idx++)
		{
			createBuffer(bufferSize, VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
				VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, m_viewUniformBuffers[idx], m_viewUniformBuffersMemory[idx]);

			vkMapMemory(device, m_viewUniformBuffersMemory[idx], 0, bufferSize, 0, &m_viewUniformBuffersMapped[idx]);
		}
	}

	void createDescriptorPool(VkDescriptorPool& pool, uint32_t samplerNum)
	{
		// each frame need one descriptor
		size_t uSize = 2; // ubo size
		std::vector<VkDescriptorPoolSize> poolSizes;
		poolSizes.resize(samplerNum + uSize); // sampler num + ubo num
		poolSizes[0].type            = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
		poolSizes[0].descriptorCount = static_cast<uint32_t>(MAX_FRAMES_IN_FLIGHT);

		poolSizes[1].type            = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
		poolSizes[1].descriptorCount = static_cast<uint32_t>(MAX_FRAMES_IN_FLIGHT);

		for (size_t idx = 0; idx < samplerNum; idx++)
		{
			poolSizes[idx + uSize].type            = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
			poolSizes[idx + uSize].descriptorCount = static_cast<uint32_t>(MAX_FRAMES_IN_FLIGHT);
		}

		VkDescriptorPoolCreateInfo poolInfo {
			.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
			.maxSets = static_cast<uint32_t>(MAX_FRAMES_IN_FLIGHT),

			.poolSizeCount = static_cast<uint32_t>(poolSizes.size()),
			.pPoolSizes = poolSizes.data(),
		};

		if (vkCreateDescriptorPool(device, &poolInfo, nullptr, &pool) != VK_SUCCESS) {
			throw std::runtime_error("failed to create descriptor pool!");
		}
	}

	void createDescriptorSets(const VkDescriptorPool& pool,
		const VkDescriptorSetLayout& descriptorSetLayout, const std::vector<VkImageView>& imageViews, const std::vector<VkSampler>& samplers,
		std::vector<VkDescriptorSet>& descriptorSets)
	{
		std::vector<VkDescriptorSetLayout> layouts(MAX_FRAMES_IN_FLIGHT, descriptorSetLayout);
		VkDescriptorSetAllocateInfo allocInfo {
			.sType              = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
			.descriptorPool     = pool,
			.descriptorSetCount = static_cast<uint32_t>(MAX_FRAMES_IN_FLIGHT),
			.pSetLayouts        = layouts.data(),
		};

		descriptorSets.resize(MAX_FRAMES_IN_FLIGHT);
		if (vkAllocateDescriptorSets(device, &allocInfo, descriptorSets.data()) != VK_SUCCESS) {
			throw std::runtime_error("failed to allocate descriptor sets!");
		}

		for (size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; i++) {
			std::vector<VkWriteDescriptorSet> descriptorWrites;
			descriptorWrites.resize(3 + imageViews.size());
			
			// ubo
			VkDescriptorBufferInfo bufferInfo {
				.buffer = m_uniformBuffers[i],
				.offset = 0,
				.range  = sizeof(UniformBufferObject),
			};
			descriptorWrites[0].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
			descriptorWrites[0].dstSet = descriptorSets[i];
			descriptorWrites[0].dstBinding = 0;
			descriptorWrites[0].dstArrayElement = 0;
			descriptorWrites[0].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
			descriptorWrites[0].descriptorCount = 1;
			descriptorWrites[0].pBufferInfo = &bufferInfo;

			// texture sampler
			std::vector<VkDescriptorImageInfo> imageInfos; // descriptorWrites 会引用每一个创建的 VkDescriptorImageInfo，所以需要用一个数组把它们存储起来
			imageInfos.resize(imageViews.size());
			for (size_t imgIdx = 0; imgIdx < imageViews.size(); imgIdx++)
			{
				VkDescriptorImageInfo imageInfo{
					.sampler     = samplers[imgIdx],
					.imageView   = imageViews[imgIdx],
					.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
				};

				// keep reference vaild
				imageInfos[imgIdx] = imageInfo;

				descriptorWrites[imgIdx + 1].sType           = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
				descriptorWrites[imgIdx + 1].dstSet          = descriptorSets[i];
				descriptorWrites[imgIdx + 1].dstBinding      = static_cast<uint32_t>(imgIdx +1);
				descriptorWrites[imgIdx + 1].dstArrayElement = 0;
				descriptorWrites[imgIdx + 1].descriptorType  = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
				descriptorWrites[imgIdx + 1].descriptorCount = 1;
				descriptorWrites[imgIdx + 1].pImageInfo      = &imageInfos[imgIdx];
			}

			// shadowMap sampler
			VkDescriptorImageInfo shadowImageInfo{
				.sampler = shadowPass.shadowMapSampler,
				.imageView = shadowPass.shadowMapImageView,
				.imageLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_READ_ONLY_OPTIMAL,
			};

			descriptorWrites[2].sType           = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
			descriptorWrites[2].dstSet          = descriptorSets[i];
			descriptorWrites[2].dstBinding      = 2 ;
			descriptorWrites[2].dstArrayElement = 0;
			descriptorWrites[2].descriptorType  = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
			descriptorWrites[2].descriptorCount = 1;
			descriptorWrites[2].pImageInfo      = &shadowImageInfo;

			// view ubo
			VkDescriptorBufferInfo viewBufferInfo{
				.buffer = m_viewUniformBuffers[i],
				.offset = 0,
				.range = sizeof(ViewUniformBufferObject),
			};
			descriptorWrites[3].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
			descriptorWrites[3].dstSet = descriptorSets[i];
			descriptorWrites[3].dstBinding = 3;
			descriptorWrites[3].dstArrayElement = 0;
			descriptorWrites[3].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
			descriptorWrites[3].descriptorCount = 1;
			descriptorWrites[3].pBufferInfo = &viewBufferInfo;

			vkUpdateDescriptorSets(device, static_cast<uint32_t>(descriptorWrites.size()), descriptorWrites.data(), 0, nullptr);
		}
	}

	VkPipelineLayout createPipelineLayout(const VkDescriptorSetLayout& dSetLayouts)
	{
		// push constant data, model matrix
		VkPushConstantRange pcRange = {
			.stageFlags = VK_SHADER_STAGE_VERTEX_BIT,
			.offset     = 0,
			.size       = sizeof(PushConstantData)
		};

		VkPipelineLayoutCreateInfo pipelineLayoutInfo{ 
			.sType                  = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
			.setLayoutCount         = 1,
			.pSetLayouts            = &dSetLayouts,
			.pushConstantRangeCount = 1, 
			.pPushConstantRanges    = &pcRange, 
		};

		VkPipelineLayout layout;
		if (vkCreatePipelineLayout(device, &pipelineLayoutInfo, nullptr, &layout) != VK_SUCCESS) {
			throw std::runtime_error("failed to create pipeline layout!");
		}
		return layout;
	}

	void createShadowMapPass()
	{
		// shadow depth texture
		shadowPass.width = SHADOW_SIZE;
		shadowPass.height = SHADOW_SIZE;

		VkFormat depthFormat = findDepthFormat();

		createImage(shadowPass.width, shadowPass.height, depthFormat, 
			VK_IMAGE_TILING_OPTIMAL, VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT,   // We will sample directly from the depth attachment for the shadow mapping
			VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, shadowPass.shadowMapImage, shadowPass.shadowMapImageMemory);

		shadowPass.shadowMapImageView = createImageView(shadowPass.shadowMapImage, depthFormat, VK_IMAGE_ASPECT_DEPTH_BIT);

		shadowPass.shadowMapSampler = createTextureSampler(
			VK_FILTER_LINEAR,
			VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE,
			VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE,
			VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE,
			VK_BORDER_COLOR_FLOAT_OPAQUE_WHITE,
			1
		);

		// ubo
		VkDeviceSize uboSize = sizeof(ShadowUBO);
		shadowPass.uniformBuffers.resize(MAX_FRAMES_IN_FLIGHT);
		shadowPass.uniformBuffersMemory.resize(MAX_FRAMES_IN_FLIGHT);
		shadowPass.uniformBuffersMapped.resize(MAX_FRAMES_IN_FLIGHT);

		for (size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; i++)
		{
			createBuffer(uboSize,
				VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
				shadowPass.uniformBuffers[i], shadowPass.uniformBuffersMemory[i]);

			vkMapMemory(device, shadowPass.uniformBuffersMemory[i], 0, uboSize, 0, &shadowPass.uniformBuffersMapped[i]);
		}
		
		// descriptor layout
		std::vector<VkDescriptorSetLayoutBinding> bindings;
		VkDescriptorSetLayoutBinding uboLayoutBind{
			.binding            = 0,
			.descriptorType     = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
			.descriptorCount    = 1,
			.stageFlags         = VK_SHADER_STAGE_VERTEX_BIT,
			.pImmutableSamplers = nullptr
		};
		bindings.push_back(uboLayoutBind);

		VkDescriptorSetLayoutCreateInfo layoutInfo{
			.sType        = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
			.bindingCount = static_cast<uint32_t>(bindings.size()),
			.pBindings    = bindings.data(),
		};

		if (vkCreateDescriptorSetLayout(device, &layoutInfo, nullptr, &shadowPass.descriptoSetLayout) != VK_SUCCESS) {
			throw std::runtime_error("failed to create descriptor set layout!");
		}

		// descriptor pool
		//createDescriptorPool(shadowPass.descriptorPool, 0);

		// each frame need one descriptor
		size_t uSize = 1; // ubo size
		std::vector<VkDescriptorPoolSize> poolSizes;
		poolSizes.resize(uSize); // sampler num + ubo num
		poolSizes[0].type = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
		poolSizes[0].descriptorCount = static_cast<uint32_t>(MAX_FRAMES_IN_FLIGHT);


		VkDescriptorPoolCreateInfo poolInfo{
			.sType         = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
			.maxSets       = static_cast<uint32_t>(MAX_FRAMES_IN_FLIGHT),

			.poolSizeCount = static_cast<uint32_t>(poolSizes.size()),
			.pPoolSizes    = poolSizes.data(),
		};

		if (vkCreateDescriptorPool(device, &poolInfo, nullptr, &shadowPass.descriptorPool) != VK_SUCCESS) {
			throw std::runtime_error("failed to create descriptor pool!");
		}


		// descriptor sets create
		std::vector<VkDescriptorSetLayout> layouts(MAX_FRAMES_IN_FLIGHT, shadowPass.descriptoSetLayout);
		VkDescriptorSetAllocateInfo allocInfo{
			.sType              = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
			.descriptorPool     = shadowPass.descriptorPool,
			.descriptorSetCount = static_cast<uint32_t>(MAX_FRAMES_IN_FLIGHT),
			.pSetLayouts        = layouts.data(),
		};

		shadowPass.descriptoSets.resize(MAX_FRAMES_IN_FLIGHT);
		if (vkAllocateDescriptorSets(device, &allocInfo, shadowPass.descriptoSets.data()) != VK_SUCCESS) {
			throw std::runtime_error("failed to allocate descriptor sets!");
		}

		// descriptor sets binding
		for (size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; i++) {
			std::vector<VkWriteDescriptorSet> descriptorWrites;
			descriptorWrites.resize(1 );

			// ubo
			VkDescriptorBufferInfo bufferInfo{
				.buffer = shadowPass.uniformBuffers[i],
				.offset = 0,
				.range = sizeof(ShadowUBO),
			};
			descriptorWrites[0].sType           = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
			descriptorWrites[0].dstSet          = shadowPass.descriptoSets[i];
			descriptorWrites[0].dstBinding      = 0;
			descriptorWrites[0].dstArrayElement = 0;
			descriptorWrites[0].descriptorType  = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
			descriptorWrites[0].descriptorCount = 1;
			descriptorWrites[0].pBufferInfo     = &bufferInfo;

			vkUpdateDescriptorSets(device, static_cast<uint32_t>(descriptorWrites.size()), descriptorWrites.data(), 0, nullptr);
		}

		// create render pass
		// depth attachment
		VkAttachmentDescription depthAttachment{
			.format         = depthFormat,
			.samples        = VK_SAMPLE_COUNT_1_BIT,
			.loadOp         = VK_ATTACHMENT_LOAD_OP_CLEAR,
			.storeOp        = VK_ATTACHMENT_STORE_OP_STORE,
			.stencilLoadOp  = VK_ATTACHMENT_LOAD_OP_DONT_CARE,
			.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE,
			.initialLayout  = VK_IMAGE_LAYOUT_UNDEFINED,
			.finalLayout    = VK_IMAGE_LAYOUT_DEPTH_STENCIL_READ_ONLY_OPTIMAL,
		};

		VkAttachmentReference depthAttachmentRef{
			.attachment     = 0, // no color
			.layout         = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL
		};

		// subpass
		VkSubpassDescription subpass{
			.pipelineBindPoint       = VK_PIPELINE_BIND_POINT_GRAPHICS,
			.inputAttachmentCount    = 0,
			.colorAttachmentCount    = 0,  // no color attachment
			//.pColorAttachments       = nullptr,
			.pDepthStencilAttachment = &depthAttachmentRef
		};

		// todo: subpass dependencies
		std::array<VkSubpassDependency, 2> dependencies {
			VkSubpassDependency{
				.srcSubpass      = VK_SUBPASS_EXTERNAL,
				.dstSubpass      = 0,
				.srcStageMask    = VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,
				.dstStageMask    = VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT,
				.srcAccessMask   = VK_ACCESS_SHADER_READ_BIT,
				.dstAccessMask   = VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT,
				.dependencyFlags = VK_DEPENDENCY_BY_REGION_BIT
			},
			VkSubpassDependency{
				.srcSubpass      = VK_SUBPASS_EXTERNAL,
				.dstSubpass      = 0,
				.srcStageMask    = VK_PIPELINE_STAGE_LATE_FRAGMENT_TESTS_BIT,
				.dstStageMask    = VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,
				.srcAccessMask   = VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT,
				.dstAccessMask   = VK_ACCESS_SHADER_READ_BIT,
				.dependencyFlags = VK_DEPENDENCY_BY_REGION_BIT
			}
		};

		std::array<VkAttachmentDescription, 1> attachments = { depthAttachment };

		VkRenderPassCreateInfo renderPassInfo{
			.sType           = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO,
			.attachmentCount = static_cast<uint32_t>(attachments.size()),
			.pAttachments    = attachments.data(),
			.subpassCount    = 1,
			.pSubpasses      = &subpass,
			.dependencyCount = static_cast<uint32_t>(dependencies.size()),
			.pDependencies   = dependencies.data(),
		};

		if (vkCreateRenderPass(device, &renderPassInfo, nullptr, &shadowPass.renderPass) != VK_SUCCESS) {
			throw std::runtime_error("failed to create render pass!");
		}


		// create framebuffer
		VkFramebufferCreateInfo framebufferInfo{
			.sType           = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO,
			.renderPass      = shadowPass.renderPass,
			.attachmentCount = 1,
			.pAttachments    = &shadowPass.shadowMapImageView,
			.width           = shadowPass.width,
			.height          = shadowPass.height,
			.layers          = 1,
		};

		if (vkCreateFramebuffer(device, &framebufferInfo, nullptr, &shadowPass.frameBuffer) != VK_SUCCESS)
		{
			throw std::runtime_error("failed to create framebuffer!");
		}

		// create pipeline layout
		shadowPass.pipelineLayout = createPipelineLayout(shadowPass.descriptoSetLayout);

		// create pipeline
		std::vector<char> vertShaderCode = readFile(shadowVertShaderPath);
		std::vector<char> fragShaderCode = readFile(shadowFragShaderPath);

		VkShaderModule vertShaderModule = createShaderModule(vertShaderCode);
		VkShaderModule fragShaderModule = createShaderModule(fragShaderCode);

		// shader stage creation
		VkPipelineShaderStageCreateInfo vertShaderStageInfo{
			.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
			.stage = VK_SHADER_STAGE_VERTEX_BIT,
			.module = vertShaderModule,
			.pName = "main",
		};

		VkPipelineShaderStageCreateInfo fragShaderStageInfo{
			.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
			.stage = VK_SHADER_STAGE_FRAGMENT_BIT,
			.module = fragShaderModule,
			.pName = "main",
		};


		VkPipelineShaderStageCreateInfo shaderStages[] = { vertShaderStageInfo, fragShaderStageInfo };

		// dynamic state
		std::vector<VkDynamicState> dynamicSates = {
			VK_DYNAMIC_STATE_VIEWPORT,
			VK_DYNAMIC_STATE_SCISSOR
		};
		VkPipelineDynamicStateCreateInfo dynamicState{
			.sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO,
			.dynamicStateCount = static_cast<uint32_t>(dynamicSates.size()),
			.pDynamicStates = dynamicSates.data()
		};


		// vertex input
		auto bindingDescription = Vertex::getBindingDescription();
		auto attributeDescriptions = Vertex::getAttributeDescription();

		VkPipelineVertexInputStateCreateInfo vertexInputInfo{
			.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO,
			.vertexBindingDescriptionCount = 1,
			.pVertexBindingDescriptions = &bindingDescription, // Optional

			.vertexAttributeDescriptionCount = static_cast<uint32_t>(attributeDescriptions.size()),
			.pVertexAttributeDescriptions = attributeDescriptions.data(), // Optional
		};


		// input assembly
		VkPipelineInputAssemblyStateCreateInfo inputAssembly{
			.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO,
			.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST,
			.primitiveRestartEnable = VK_FALSE,
		};


		// viewport and scissors, image to framebuffer
		VkViewport viewport{
			.x = 0.0f,
			.y = 0.0f,
			.width = (float)swapChainExtent.width,
			.height = (float)swapChainExtent.height,
			.minDepth = 0.0f, // [0.0f, 1.0f]
			.maxDepth = 1.0f, // [0.0f, 1.0f]
		};

		// which regions pixels will actually be stored. 
		VkRect2D scissor{
			.offset = { 0, 0 },
			.extent = swapChainExtent,
		};

		VkPipelineViewportStateCreateInfo viewportState{
			.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO,
			.viewportCount = 1,
			.pViewports = &viewport,
			.scissorCount = 1,
			.pScissors = &scissor,
		};


		// rasterization
		VkPipelineRasterizationStateCreateInfo rasterizer{
			.sType                   = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO,
			.depthClampEnable        = VK_FALSE,
			.rasterizerDiscardEnable = VK_FALSE, // fragment to framebuffer
			.polygonMode             = VK_POLYGON_MODE_FILL,
			.cullMode                = VK_CULL_MODE_NONE,
			.frontFace               = VK_FRONT_FACE_COUNTER_CLOCKWISE,
			.depthBiasEnable         = VK_FALSE,
			.depthBiasConstantFactor = 0.0f, // Optional
			.depthBiasClamp          = 0.0f, // Optional
			.depthBiasSlopeFactor    = 0.0f, // Optional
			.lineWidth               = 1.0f,
		};

		VkPipelineMultisampleStateCreateInfo multisampling{
			.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO,
			.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT,
			.sampleShadingEnable = VK_FALSE,
			.minSampleShading = 1.0f, // Optional
			.pSampleMask = nullptr, // Optional
			.alphaToCoverageEnable = VK_FALSE, // Optional
			.alphaToOneEnable = VK_FALSE, // Optional
		};

		// color blending
		VkPipelineColorBlendAttachmentState colorBlendAttachment{
			.blendEnable         = VK_FALSE,
			.srcColorBlendFactor = VK_BLEND_FACTOR_ONE, // Optional
			.dstColorBlendFactor = VK_BLEND_FACTOR_ZERO, // Optional
			.colorBlendOp        = VK_BLEND_OP_ADD, // Optional
			.srcAlphaBlendFactor = VK_BLEND_FACTOR_ONE, // Optional
			.dstAlphaBlendFactor = VK_BLEND_FACTOR_ZERO, // Optional
			.alphaBlendOp        = VK_BLEND_OP_ADD, // Optional
			.colorWriteMask      = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT | VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT,
		};

		VkPipelineColorBlendStateCreateInfo colorBlending{
			.sType           = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO,
			.logicOpEnable   = VK_FALSE,
			.logicOp         = VK_LOGIC_OP_COPY, // Optional
			.attachmentCount = 0,
			.pAttachments    = &colorBlendAttachment,
			.blendConstants  = { 0.0f }, // Optional
		};

		// depth and stencil
		VkPipelineDepthStencilStateCreateInfo depthStencil{
			.sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO,
			.depthTestEnable = VK_TRUE,
			.depthWriteEnable = VK_TRUE,
			.depthCompareOp = VK_COMPARE_OP_LESS_OR_EQUAL,
			.depthBoundsTestEnable = VK_FALSE,
			.stencilTestEnable = VK_FALSE,
			.front = {},
			.back = {},
			.minDepthBounds = 0.0f, // Optional
			.maxDepthBounds = 1.0f, // Optional
		};

		// create pipeline
		VkGraphicsPipelineCreateInfo pipelineInfo{
			.sType               = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO,
			.stageCount          = 1,
			.pStages             = shaderStages,

			.pVertexInputState   = &vertexInputInfo,
			.pInputAssemblyState = &inputAssembly,
			.pViewportState      = &viewportState,
			.pRasterizationState = &rasterizer,
			.pMultisampleState   = &multisampling,
			.pDepthStencilState  = &depthStencil, // Optional
			.pColorBlendState    = &colorBlending,
			.pDynamicState       = &dynamicState,

			.layout              = shadowPass.pipelineLayout,

			.renderPass          = shadowPass.renderPass,
			.subpass             = 0,

			.basePipelineHandle  = VK_NULL_HANDLE, // Optional
			.basePipelineIndex   = -1, // Optional
		};

		if (vkCreateGraphicsPipelines(device, VK_NULL_HANDLE, 1, &pipelineInfo, nullptr, &shadowPass.pipeline) != VK_SUCCESS) {
			throw std::runtime_error("failed to create graphics pipeline!");
		}

		// clean up resources
		vkDestroyShaderModule(device, fragShaderModule, nullptr);
		vkDestroyShaderModule(device, vertShaderModule, nullptr);

	}

	void createSenceRenderPass()
	{
		uint32_t samplerNums = 1; // texture count, only base color no metarials
		// create decsriptor layout
		baseScenePass.descriptorSetLayout = createDescriptorSetLayout(samplerNums);

		// create pipeline layout 
		baseScenePass.pipelineLayout = createPipelineLayout(baseScenePass.descriptorSetLayout);

		// create pipline
		baseScenePass.pipeline = createGraphicsPipeline(baseScenePass.pipelineLayout);

		// load Model and create resource
		auto createObjectResource = [this](RenderObject& model, const std::string& modelPath, const std::vector<std::string>& texImages) {
			// model 
			loadModel(modelPath, model);
			model.textureImages.resize(texImages.size());
			model.textureImageViews.resize(texImages.size());
			model.textureImageMemorys.resize(texImages.size());
			model.textureSamplers.resize(texImages.size());

			//texture
			for (size_t idx = 0; idx < texImages.size(); idx++)
			{
				// create texture image
				createTextureImage(texImages[idx], model.textureImages[idx], model.textureImageMemorys[idx]);

				// create image view
				model.textureImageViews[idx] = createTextureImageView(model.textureImages[idx]);

				// create sampler
				model.textureSamplers[idx] = createTextureSampler(
					VK_FILTER_LINEAR,
					VK_SAMPLER_ADDRESS_MODE_REPEAT,
					VK_SAMPLER_ADDRESS_MODE_REPEAT,
					VK_SAMPLER_ADDRESS_MODE_REPEAT,
					VK_BORDER_COLOR_INT_OPAQUE_BLACK,
					0
				);
			};

			// buffer
			createVertexBuffer(model.vertices, model.vertexBuffer, model.vertexBufferMemory);
			createIndexBuffer(model.indices, model.indexBuffer, model.indexBufferMemory);

			createDescriptorPool(model.descriptorPool, static_cast<uint32_t>(texImages.size() + 1));
			createDescriptorSets(model.descriptorPool, baseScenePass.descriptorSetLayout, model.textureImageViews, model.textureSamplers, model.descriptorSets);
		};

		// models in scene and resource
		RenderObject stage;
		std::string stageModelPath = "../../resources/models/stage.obj";
		std::vector<std::string> stageTexImgs{ "../../resources/texture/stage.png" };
		// create resouce of vertex, index, texture, descriptor 
		createObjectResource(stage, stageModelPath, stageTexImgs);
		stage.modelMatrix = glm::rotate(stage.modelMatrix, glm::radians(-90.0f), glm::vec3(1.0f, 0.0f, 0.0f));
		stage.modelMatrix = glm::translate(stage.modelMatrix, glm::vec3(0.0f, 0.0f, -1.3f));
		stage.modelMatrix = glm::scale(stage.modelMatrix, glm::vec3(2.4f, 2.2f, 2.4f ));
		baseScenePass.renderObjects.push_back(stage);

		RenderObject marry;
		std::string marryModelPath = "../../resources/models/Marry.obj";
		std::vector<std::string> marryTexImgs{ "../../resources/texture/MC003_Kozakura_Mari.png" };
		createObjectResource(marry, marryModelPath, marryTexImgs);

		marry.modelMatrix = glm::translate(marry.modelMatrix, glm::vec3(0.0f, -1.2f, 0.0f));
		marry.modelMatrix = glm::scale(marry.modelMatrix, glm::vec3(0.8f, 0.8f, 0.8f));

		baseScenePass.renderObjects.push_back(marry);
	}
	
	void createCommandBuffer()
	{
		commandBuffers.resize(MAX_FRAMES_IN_FLIGHT);
		VkCommandBufferAllocateInfo allocInfo{};
		allocInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
		allocInfo.commandPool = commandPool;
		allocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
		allocInfo.commandBufferCount = (uint32_t)commandBuffers.size();

		if (vkAllocateCommandBuffers(device, &allocInfo, commandBuffers.data()) != VK_SUCCESS) {
			throw std::runtime_error("failed to allocate command buffers!");
		}
	}

	void recordCommandBuffer(VkCommandBuffer commandBuffer, uint32_t imageIndex)
	{
		VkCommandBufferBeginInfo beginInfo {
			.sType            = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
			.flags            = 0, // Optional
			.pInheritanceInfo = nullptr, // Optional
		};

		if (vkBeginCommandBuffer(commandBuffer, &beginInfo) != VK_SUCCESS) {
			throw std::runtime_error("failed to begin recording command buffer!");
		}
		float time = glfwGetTime();
		// shadow pass
		{
			std::array<VkClearValue, 1> clearColors{};
			clearColors[0].depthStencil = { 1.0f, 0 };
			VkRenderPassBeginInfo renderPassInfo{
				.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO,
				.renderPass = shadowPass.renderPass,
				.framebuffer = shadowPass.frameBuffer,

				.renderArea = {
					.offset = { 0, 0 },
					.extent = {
						.width = shadowPass.width,
						.height = shadowPass.height
					}
				},

				.clearValueCount = static_cast<uint32_t>(clearColors.size()),
				.pClearValues = clearColors.data(),
			};

			// start render pass
			vkCmdBeginRenderPass(commandBuffer, &renderPassInfo, VK_SUBPASS_CONTENTS_INLINE);

			// scene viewport
			VkViewport viewport{
				.x = 0.0f,
				.y = 0.0f,
				.width = static_cast<float>(shadowPass.width),
				.height = static_cast<float>(shadowPass.height),
				.minDepth = 0.0f,
				.maxDepth = 1.0f,
			};
			vkCmdSetViewport(commandBuffer, 0, 1, &viewport);

			VkRect2D scissor{
				.offset = { 0, 0 },
				.extent = {
					.width = shadowPass.width,
					.height = shadowPass.height
				}
			};
			vkCmdSetScissor(commandBuffer, 0, 1, &scissor);


			for (size_t idx = 0; idx < baseScenePass.renderObjects.size(); idx++)
			{
				const auto& renderObject = baseScenePass.renderObjects[idx];

				//updateUniformBuffer(currentFrame, renderObject.modelMatrix, idx);
				PushConstantData pcData = { renderObject.modelMatrix };
				if (idx == 1)
					pcData.modelMatrix = glm::rotate(pcData.modelMatrix, time * glm::radians(90.0f), glm::vec3(0.0f, 1.0f, 0.0f));
				vkCmdPushConstants(commandBuffer, shadowPass.pipelineLayout, VK_SHADER_STAGE_VERTEX_BIT, 0, sizeof(PushConstantData), &pcData);

				// binding pipeline
				vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, shadowPass.pipeline);

				// binding vertex buffer 
				VkBuffer vertexBuffers[] = { renderObject.vertexBuffer };
				VkDeviceSize offsets[] = { 0 };
				vkCmdBindVertexBuffers(commandBuffer, 0, 1, vertexBuffers, offsets);

				// binding index buffer
				vkCmdBindIndexBuffer(commandBuffer, renderObject.indexBuffer, 0, VK_INDEX_TYPE_UINT32);

				// binding descriptor set
				vkCmdBindDescriptorSets(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, shadowPass.pipelineLayout, 0, 1, &shadowPass.descriptoSets[currentFrame], 0, nullptr);

				// draw index
				vkCmdDrawIndexed(commandBuffer, static_cast<uint32_t>(renderObject.indices.size()), 1, 0, 0, 0);
			}

			// end render pass
			vkCmdEndRenderPass(commandBuffer);
		}

		// lighting pass
		{
			// clear value for different attachment
			std::array<VkClearValue, 2> clearColors{};
			clearColors[0].color = { {0.2f, 0.2f, 0.2f, 1.0f} };
			clearColors[1].depthStencil = { 1.0f, 0 };

			// start render scene pass
			VkRenderPassBeginInfo renderPassInfo{
				.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO,
				.renderPass = renderPass,
				.framebuffer = swapChainFramebuffers[imageIndex],

				.renderArea = {
					.offset = { 0, 0 },
					.extent = swapChainExtent,
				},

				.clearValueCount = static_cast<uint32_t>(clearColors.size()),
				.pClearValues = clearColors.data(),
			};

			// start render pass
			vkCmdBeginRenderPass(commandBuffer, &renderPassInfo, VK_SUBPASS_CONTENTS_INLINE);

			// scene viewport
			VkViewport viewport{
				.x = 0.0f,
				.y = 0.0f,
				.width = static_cast<float>(swapChainExtent.width),
				.height = static_cast<float>(swapChainExtent.height),
				.minDepth = 0.0f,
				.maxDepth = 1.0f,
			};
			vkCmdSetViewport(commandBuffer, 0, 1, &viewport);

			VkRect2D scissor{
				.offset = { 0, 0 },
				.extent = swapChainExtent
			};
			vkCmdSetScissor(commandBuffer, 0, 1, &scissor);


			for (size_t idx = 0; idx < baseScenePass.renderObjects.size(); idx++)
			{
				const auto& renderObject = baseScenePass.renderObjects[idx];

				//updateUniformBuffer(currentFrame, renderObject.modelMatrix, idx);
				PushConstantData pcData = { renderObject.modelMatrix };
				if (idx == 1)
					pcData.modelMatrix = glm::rotate(pcData.modelMatrix, time * glm::radians(90.0f), glm::vec3(0.0f, 1.0f, 0.0f));
				vkCmdPushConstants(commandBuffer, baseScenePass.pipelineLayout, VK_SHADER_STAGE_VERTEX_BIT, 0, sizeof(PushConstantData), &pcData);

				// binding pipeline
				vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, baseScenePass.pipeline);

				// binding vertex buffer 
				VkBuffer vertexBuffers[] = { renderObject.vertexBuffer };
				VkDeviceSize offsets[] = { 0 };
				vkCmdBindVertexBuffers(commandBuffer, 0, 1, vertexBuffers, offsets);

				// binding index buffer
				vkCmdBindIndexBuffer(commandBuffer, renderObject.indexBuffer, 0, VK_INDEX_TYPE_UINT32);

				// binding descriptor set
				vkCmdBindDescriptorSets(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, baseScenePass.pipelineLayout, 0, 1, &renderObject.descriptorSets[currentFrame], 0, nullptr);

				// draw index
				vkCmdDrawIndexed(commandBuffer, static_cast<uint32_t>(renderObject.indices.size()), 1, 0, 0, 0);
			}

			// end render pass
			vkCmdEndRenderPass(commandBuffer);
		}

		if (vkEndCommandBuffer(commandBuffer) != VK_SUCCESS) {
			throw std::runtime_error("failed to record command buffer!");
		}
	}

	void updateLight()
	{
		// Animate the light source
		float rad = 5;
		gInput.lightPos.x = -6.0f * cos(glm::radians(glfwGetTime() * 30.0f)) ;
		//gInput.lightPos.y = 50.0f + sin(glm::radians(glfwGetTime() * 360.0f)) * 20.0f;
		gInput.lightPos.z = 5.0f * sin(glm::radians(glfwGetTime() * 30.0f));
	}

	void updateUniformBuffer(uint32_t currentImage)
	{
		static auto startTime = std::chrono::high_resolution_clock::now();

		auto currentTime = std::chrono::high_resolution_clock::now();
		float time = std::chrono::duration<float, std::chrono::seconds::period>(currentTime - startTime).count();

		updateLight();

		// shadow
		ShadowUBO shadowUbo{};
		shadowUbo.view = glm::lookAt(gInput.lightPos, gInput.target, gInput.cameraUp);
		shadowUbo.proj = glm::ortho(-10.0f, 10.0f, -10.0f, 10.0f, gInput.zNear, gInput.zFar);
		//shadowUbo.proj = glm::perspective(glm::radians(gInput.lightFOV), 1.0f, gInput.zNear, gInput.zFar);
		shadowUbo.proj[1][1] *= -1;

		memcpy(shadowPass.uniformBuffersMapped[currentImage], &shadowUbo, sizeof(shadowUbo));


		// VP 
		UniformBufferObject ubo{};

		ubo.view = glm::lookAt(gInput.cameraPos, gInput.target, gInput.cameraUp);
		ubo.proj = glm::perspective(glm::radians(gInput.lightFOV), swapChainExtent.width / (float)swapChainExtent.height, gInput.zNear, gInput.zFar);
		// flip the sign on the scaling factor of the Y axis in the projection matrix
		ubo.proj[1][1] *= -1;

		ubo.lightVP = shadowUbo.proj * shadowUbo.view;

		// update memory
		memcpy(m_uniformBuffersMapped[currentImage], &ubo, sizeof(ubo));

		ViewUniformBufferObject vubo{};
		vubo.cameraPos = glm::vec4(gInput.cameraPos, 1.0);
		vubo.lightPos = glm::vec4(gInput.lightPos, 0.0f);

		// update memory
		memcpy(m_viewUniformBuffersMapped[currentImage], &vubo, sizeof(vubo));

	}
	
	void drawFrame()
	{
		// wait for previous frame
		vkWaitForFences(device, 1, &inFlightFences[currentFrame], VK_TRUE, UINT64_MAX);

		// acquire an image from swap chain
		uint32_t imageIndex;
		VkResult result = vkAcquireNextImageKHR(device, swapChain, UINT64_MAX, imageAvailableSemaphores[currentFrame], VK_NULL_HANDLE, &imageIndex);
		if (result == VK_ERROR_OUT_OF_DATE_KHR)
		{
			recreateSwapChain();
			return;
		}
		else if (result != VK_SUCCESS && result != VK_SUBOPTIMAL_KHR)
		{
			throw std::runtime_error("failed to acquire swap chain image!");
		}

		// update ubo 
		updateUniformBuffer(currentFrame);

		// only reset the fence if we are submitting work
		vkResetFences(device, 1, &inFlightFences[currentFrame]); // reset states


		// record command buffer, make sure previous command buffer finished execute
		vkResetCommandBuffer(commandBuffers[currentFrame], 0);
		recordCommandBuffer(commandBuffers[currentFrame], imageIndex);

		// submit the command buffer
		VkSemaphore waitSemaphores[]      = { imageAvailableSemaphores[currentFrame] };
		VkPipelineStageFlags waitStages[] = { VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT };
		VkSemaphore signalSemaphores[]    = { renderFinishedSemaphores[currentFrame] };
		
		VkSubmitInfo submitInfo {
			.sType                = VK_STRUCTURE_TYPE_SUBMIT_INFO,

			.waitSemaphoreCount   = 1,
			.pWaitSemaphores      = waitSemaphores,
			.pWaitDstStageMask    = waitStages,

			.commandBufferCount   = 1,
			.pCommandBuffers      = &commandBuffers[currentFrame],

			.signalSemaphoreCount = 1,
			.pSignalSemaphores    = signalSemaphores,
		};

		if (vkQueueSubmit(graphicsQueue, 1, &submitInfo, inFlightFences[currentFrame]) != VK_SUCCESS)
		{
			throw std::runtime_error("failed to submit draw command buffer!");
		}

		// presentation
		VkSwapchainKHR swapChains[] = { swapChain };

		VkPresentInfoKHR presentInfo {
			.sType              = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR,
			.waitSemaphoreCount = 1,
			.pWaitSemaphores    = signalSemaphores,

			.swapchainCount     = 1,
			.pSwapchains        = swapChains,
			.pImageIndices      = &imageIndex,

			.pResults           = nullptr // Optional
		};

		result = vkQueuePresentKHR(presentQueue, &presentInfo);
		if (result == VK_ERROR_OUT_OF_DATE_KHR || result == VK_SUBOPTIMAL_KHR || framebufferResized)
		{
			framebufferResized = false;
			recreateSwapChain();
		}
		else if (result != VK_SUCCESS)
		{
			throw std::runtime_error("failed to present swap chain image!");
		}

		// advance to next frame
		currentFrame = (currentFrame + 1) % MAX_FRAMES_IN_FLIGHT;
	}

	void createSyncObjects()
	{
		imageAvailableSemaphores.resize(MAX_FRAMES_IN_FLIGHT);
		renderFinishedSemaphores.resize(MAX_FRAMES_IN_FLIGHT);
		inFlightFences.resize(MAX_FRAMES_IN_FLIGHT);

		VkSemaphoreCreateInfo semaphoreInfo{};
		semaphoreInfo.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;

		VkFenceCreateInfo fenceInfo{};
		fenceInfo.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
		fenceInfo.flags = VK_FENCE_CREATE_SIGNALED_BIT;

		for (size_t idx = 0; idx < MAX_FRAMES_IN_FLIGHT; idx++)
		{

			if (vkCreateSemaphore(device, &semaphoreInfo, nullptr, &imageAvailableSemaphores[idx]) != VK_SUCCESS
				|| vkCreateSemaphore(device, &semaphoreInfo, nullptr, &renderFinishedSemaphores[idx]) != VK_SUCCESS
				|| vkCreateFence(device, &fenceInfo, nullptr, &inFlightFences[idx]) != VK_SUCCESS
				)
			{
				throw std::runtime_error("failed to create semaphores!");
			}
		}
	}

	void cleanupSwapChain()
	{
		vkDestroyImageView(device, depthImageView, nullptr);
		vkDestroyImage(device, depthImage, nullptr);
		vkFreeMemory(device, depthImageMemory, nullptr);

		for (size_t i = 0; i < swapChainFramebuffers.size(); i++) {
			vkDestroyFramebuffer(device, swapChainFramebuffers[i], nullptr);
		}

		for (size_t i = 0; i < swapChainImageViews.size(); i++) {
			vkDestroyImageView(device, swapChainImageViews[i], nullptr);
		}

		vkDestroySwapchainKHR(device, swapChain, nullptr);
	}

	void recreateSwapChain()
	{
		// handle minimization
		int width = 0, height = 0;
		glfwGetFramebufferSize(window, &width, &height);
		std::cout << "Recreate swap chain, width:" << width << ", height:" << height <<"\n";
		while (width == 0 || height == 0)
		{
			if (glfwWindowShouldClose(window))
				return;
			glfwGetFramebufferSize(window, &width, &height);
			glfwWaitEvents(); // pause until window not minimization
		}

		// ensure all resources not in use
		vkDeviceWaitIdle(device);

		cleanupSwapChain();

		creataSwapChain();
		createSwapChainImageViews();
		createDepthResources();
		createFramebuffers();
	}

	void cleanup()
	{
		cleanupSwapChain();
		// destroy shaowMap pass resource
		vkDestroyRenderPass(device, shadowPass.renderPass, nullptr);
		vkDestroyFramebuffer(device, shadowPass.frameBuffer, nullptr);
		vkDestroyDescriptorPool(device, shadowPass.descriptorPool, nullptr);
		vkDestroyDescriptorSetLayout(device, shadowPass.descriptoSetLayout, nullptr);
		vkDestroyPipelineLayout(device, shadowPass.pipelineLayout, nullptr);
		vkDestroyPipeline(device, shadowPass.pipeline, nullptr);
		vkDestroyImageView(device, shadowPass.shadowMapImageView, nullptr);
		vkDestroySampler(device, shadowPass.shadowMapSampler, nullptr);
		vkDestroyImage(device, shadowPass.shadowMapImage, nullptr);
		vkFreeMemory(device, shadowPass.shadowMapImageMemory, nullptr);
		for (size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; i++)
		{
			vkDestroyBuffer(device, shadowPass.uniformBuffers[i], nullptr);
			vkFreeMemory(device, shadowPass.uniformBuffersMemory[i], nullptr);
		}

		// destroy base scene resource
		vkDestroyPipeline(device, baseScenePass.pipeline, nullptr);
		vkDestroyPipelineLayout(device, baseScenePass.pipelineLayout, nullptr);
		vkDestroyDescriptorSetLayout(device, baseScenePass.descriptorSetLayout, nullptr);
		for (size_t i = 0; i < baseScenePass.renderObjects.size(); i++)
		{
			auto& object = baseScenePass.renderObjects[i];
			vkDestroyDescriptorPool(device, object.descriptorPool, nullptr);

			for (size_t j = 0; j < object.textureImages.size(); j++)
			{
				vkDestroyImageView(device, object.textureImageViews[j], nullptr);
				vkDestroySampler(device, object.textureSamplers[j], nullptr);
				vkDestroyImage(device, object.textureImages[j], nullptr);
				vkFreeMemory(device, object.textureImageMemorys[j], nullptr);
			}

			vkDestroyBuffer(device, object.vertexBuffer, nullptr);
			vkFreeMemory(device, object.vertexBufferMemory, nullptr);

			vkDestroyBuffer(device, object.indexBuffer, nullptr);
			vkFreeMemory(device, object.indexBufferMemory, nullptr);
		}

		for (size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; i++) {
			vkDestroyBuffer(device, m_uniformBuffers[i], nullptr);
			vkFreeMemory(device, m_uniformBuffersMemory[i], nullptr);

			vkDestroyBuffer(device, m_viewUniformBuffers[i], nullptr);
			vkFreeMemory(device, m_viewUniformBuffersMemory[i], nullptr);
		}

		for (size_t idx = 0; idx < MAX_FRAMES_IN_FLIGHT; idx++)
		{
			vkDestroySemaphore(device, imageAvailableSemaphores[idx], nullptr);
			vkDestroySemaphore(device, renderFinishedSemaphores[idx], nullptr);
			vkDestroyFence(device, inFlightFences[idx], nullptr);
		}

		vkDestroyCommandPool(device, commandPool, nullptr);

		vkDestroyRenderPass(device, renderPass, nullptr);

		vkDestroyDevice(device, nullptr);

		if (enbleValidationLayers)
		{
			DestroyDebugUtilsMessengerEXT(instance, debugMessenger, nullptr);
		}

		vkDestroySurfaceKHR(instance, surface, nullptr);
		vkDestroyInstance(instance, nullptr);

		glfwDestroyWindow(window);

		glfwTerminate();
	}


	static void framebufferResizeCallback(GLFWwindow* window, int width, int height) {
		auto app = reinterpret_cast<HelloTriangle*>(glfwGetWindowUserPointer(window));
		app->framebufferResized = true;
	}

private:
	GLFWwindow* window;
	VkInstance instance;
	VkDebugUtilsMessengerEXT debugMessenger;

	VkSurfaceKHR surface;

	VkPhysicalDevice physicalDevice = VK_NULL_HANDLE;

	// logic device
	VkDevice device;
	
	// queue
	VkQueue graphicsQueue;
	VkQueue presentQueue;

	// swap chain
	VkSwapchainKHR swapChain;
	VkFormat swapChainImageFormat;
	VkExtent2D swapChainExtent;
	std::vector<VkImage> swapChainImages;
	std::vector<VkImageView> swapChainImageViews;


	VkRenderPass renderPass;

	std::vector<VkFramebuffer> swapChainFramebuffers;

	VkCommandPool commandPool;

	/* each frame has their own */ 
	std::vector<VkCommandBuffer> commandBuffers;

	std::vector<VkSemaphore> imageAvailableSemaphores;
	std::vector<VkSemaphore> renderFinishedSemaphores;
	std::vector<VkFence> inFlightFences;

	bool framebufferResized = false;

	/* cuurent frame index */
	uint32_t currentFrame = 0;

	// all scene resource
	ShadowMapPass shadowPass;
	BaseScenePass baseScenePass;


	std::vector<VkBuffer> m_uniformBuffers;
	std::vector<VkDeviceMemory> m_uniformBuffersMemory;
	std::vector<void*> m_uniformBuffersMapped;

	std::vector<VkBuffer> m_viewUniformBuffers;
	std::vector<VkDeviceMemory> m_viewUniformBuffersMemory;
	std::vector<void*> m_viewUniformBuffersMapped;

	VkImage depthImage;
	VkDeviceMemory depthImageMemory;
	VkImageView depthImageView;
};


int main() {
    HelloTriangle app;

	try
	{
		app.run();
	}
	catch (const std::exception& e)
	{
		std::cerr << e.what() << '\n';
		return EXIT_FAILURE;
	}

    return EXIT_SUCCESS;
}