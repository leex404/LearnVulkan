#version 450

layout(location = 0) in vec3 fragPos;
layout(location = 1) in vec3 fragColor;
layout(location = 2) in vec3 fragNormal;
layout(location = 3) in vec2 fragTexCoord;
layout(location = 4) in vec4 fragPosFromLight;

layout(location = 0) out vec4 outColor;

layout(binding = 1) uniform sampler2D texSampler;
layout(binding = 2) uniform sampler2D shadowMap;



layout (binding  = 3) uniform UniformBufferObject {
	vec4 uCameraPos;
    vec4 uLightPos;
};

vec3 uLightColor = {1.0f, 1.0f, 1.0f};

float zNear = 0.1f;
float zFar = 15.0f;
float lightWidth = 1.0f;



// base light color
vec3 blinnPhong(float shadow)
{
	vec3 baseColor = texture(texSampler, fragTexCoord).rgb;

	vec3 norm = normalize(fragNormal);
	vec3 viewDir = normalize(uCameraPos.xyz - fragPos);
	vec3 lightDir = normalize(uLightPos.xyz - fragPos);

	// ambient 
	vec3 ambient = 0.3f * uLightColor;

	// diffuse
	float kd = max(dot(lightDir, norm), 0.0f);
	vec3 diffuse = kd * uLightColor;

	// specular
	vec3 halfv = normalize(lightDir + viewDir);
	float kp = pow( max( dot(halfv, norm), 0.0f), 64.0f);
	vec3 specular = kp * uLightColor;

	return (ambient + (1.0 - shadow)*(diffuse + specular)) * baseColor;
}

vec3 phong(float shadow)
{
	vec3 baseColor = texture(texSampler, fragTexCoord).rgb;
	
	vec3 norm = normalize(fragNormal);
    vec3 viewDir = normalize(uCameraPos.xyz - fragPos);
    vec3 lightDir = normalize(uLightPos.xyz - fragPos);

    // ambient
    vec3 ambient = 0.3f * uLightColor;

    // diffuse
    float kd  = max( dot(lightDir, norm), 0.0);
    vec3 diffuse = kd * uLightColor;

    // specular
    float spec = 0.05f;
    vec3 reflectDir = reflect(-lightDir, norm);
    float kp = pow( max( dot(viewDir, reflectDir), 0.0f), 32.0f );
    vec3 specular = spec * kp * uLightColor;
	
	return (ambient + (1.0 - shadow) * (diffuse + specular)) * baseColor;
}


// only shadowMap simple
float hardShadow(vec3 coord, float bias)
{
	float closestDepth = texture(shadowMap, coord.xy).r; 
    float currentDepth = coord.z;

    float shadow = currentDepth - bias > closestDepth  ? 1.0 : 0.0;
	if(coord.z > 1.0)
       shadow = 0.0;
	return shadow;
}

// Percentage Closer Filtering (PCF)
float PCF(vec3 coord, float bias, float radius )
{
	float closestDepth = texture(shadowMap, coord.xy).r; 
    float currentDepth = coord.z;

	int count = 0;
	int step = 2;
	float shadow = 0.0;
    vec2  texelSize = 1.0 / vec2(textureSize(shadowMap, 0));
    for(int x = -step; x <= step; ++x) {
        for(int y = -step; y <= step; ++y) {
            float pcfDepth = texture(shadowMap, coord.xy + radius * vec2(x, y) * texelSize).r; 
            shadow += currentDepth - bias > pcfDepth  ? 1.0 : 0.0;   
			count++;
        }    
    }
    shadow /= float(count);
    
    // keep the shadow at 0.0 when outside the far_plane region of the light's frustum.
    if(coord.z > 1.0 ) {
       shadow = 0.0;
    }

	return shadow;
}


// Percentage Closer Soft Shadows (PCSS) 
float findBlocker(vec2 uv, float zReceiver) 
{
    int blockerNum = 0;
    float blockDepth = 0.0;
    float radius = lightWidth * (zReceiver - zNear / zFar) / zReceiver;;

    int step = 1;
    vec2  texelSize = 1.0 / vec2(textureSize(shadowMap, 0));
    for(int x = -step; x <= step; ++x) {
        for(int y = -step; y <= step; ++y) {
            float shadowMapDepth = texture(shadowMap, uv + radius*vec2(x, y) * texelSize).r;
            if(zReceiver > shadowMapDepth) 
            {
                blockDepth += shadowMapDepth;
                ++blockerNum;
            }
        }
    }

    if(blockerNum == 0) {
        return 1.0;
    }
  
    return blockDepth / float(blockerNum);
}

float PCSS(vec3 coord, float bias){
  float zReceiver = coord.z;

  // STEP 1: blocker search to get avgblocker depth
  float avgBlockerDepth = findBlocker(coord.xy, zReceiver);

  // STEP 2: penumbra estimation
  float penumbra = (zReceiver - avgBlockerDepth) * lightWidth / avgBlockerDepth;
  float filterRadiusUV = penumbra;

  // STEP 3: PCF filtering
  return PCF(coord, bias, filterRadiusUV);
}

void main() {
	
	// to NDC
	vec3 shadowCoord = fragPosFromLight.xyz / fragPosFromLight.w;
	
    float bias = 0.0f;
    vec3 lightDir = normalize(uLightPos.xyz - fragPos);
    bias = max(0.035 * (1.0 - dot(fragNormal, lightDir)), 0.005);

	//shadowCoord.y = 1.0 - shadowCoord.y;

	//float shadow = hardShadow(shadowCoord, bias);
	// float shadow = PCF(shadowCoord, bias, 0.0);
    float shadow = PCSS(shadowCoord, bias);

    outColor = vec4(blinnPhong(shadow), 1.0);
	// float closestDepth = texture(shadowMap, shadowCoord.xy).r;
    // outColor = vec4( 1.0 - (1.0 - closestDepth) * 100.0);
}