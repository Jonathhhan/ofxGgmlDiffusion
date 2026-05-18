#include "ofApp.h"
#if __has_include("ofxGgmlDiffusionNativeBackend.h")
#include "ofxGgmlDiffusionNativeBackend.h"
#elif __has_include("ofxGgmlDiffusion/ofxGgmlDiffusionNativeBackend.h")
#include "ofxGgmlDiffusion/ofxGgmlDiffusionNativeBackend.h"
#else
#error "Cannot find ofxGgmlDiffusionNativeBackend.h. Ensure ofxGgmlDiffusion is linked correctly."
#endif

#include "imgui.h"

#include <algorithm>
#include <array>
#include <cstdlib>
#include <limits>
#include <sstream>
#include <utility>
#include <vector>

namespace {
const char* schedulerNames[] = {
	"Auto",
	"Default",
	"LCM",
	"Turbo",
	"Flow match"
};

std::string getEnvironmentVariable(const std::string& name) {
#if defined(_MSC_VER)
	char* value = nullptr;
	size_t valueSize = 0;
	if (_dupenv_s(&value, &valueSize, name.c_str()) != 0 || value == nullptr) {
		return "";
	}
	std::string result(value);
	std::free(value);
	return result;
#else
	const char* value = std::getenv(name.c_str());
	return value != nullptr ? value : "";
#endif
}

bool isNativeDiffusionBackendEnabled() {
	const auto capabilities = ofxGgmlDiffusionGetNativeCapabilities();
	return capabilities.stableDiffusionEnabled;
}

std::string getNativeBackendSetupHint() {
	return "stable-diffusion.cpp is opt-in. Run ..\\scripts\\build-stable-diffusion.bat, then regenerate this example project.";
}

int schedulerToIndex(ofxGgmlDiffusionScheduler scheduler) {
	return static_cast<int>(scheduler);
}

ofxGgmlDiffusionScheduler indexToScheduler(int index) {
	if (index < 0 || index > static_cast<int>(ofxGgmlDiffusionScheduler::FlowMatch)) {
		return ofxGgmlDiffusionScheduler::Auto;
	}
	return static_cast<ofxGgmlDiffusionScheduler>(index);
}

int clampToMultipleOf64(int value) {
	const int clamped = std::clamp(value, 64, 2048);
	return std::max(64, (clamped / 64) * 64);
}

void writeBuffer(std::array<char, 2048>& buffer, const std::string& value) {
	std::fill(buffer.begin(), buffer.end(), '\0');
	std::snprintf(buffer.data(), buffer.size(), "%s", value.c_str());
}

void writeBuffer(std::array<char, 1024>& buffer, const std::string& value) {
	std::fill(buffer.begin(), buffer.end(), '\0');
	std::snprintf(buffer.data(), buffer.size(), "%s", value.c_str());
}

bool isDiffusionModelExtension(const std::string& extension) {
	const auto lowerExtension = ofToLower(extension);
	return lowerExtension == "safetensors" ||
		lowerExtension == "gguf" ||
		lowerExtension == "ckpt";
}

bool isLikelyAuxiliaryModel(const std::string& fileName) {
	const auto lowerName = ofToLower(fileName);
	const std::vector<std::string> auxiliaryMarkers = {
		"clip",
		"control",
		"esrgan",
		"lora",
		"photomaker",
		"taesd",
		"text_encoder",
		"upscaler",
		"vae"
	};
	for (const auto& marker : auxiliaryMarkers) {
		if (lowerName.find(marker) != std::string::npos) {
			return true;
		}
	}
	return false;
}

std::string findFirstDiffusionModelInDirectory(const std::string& directoryPath) {
	ofDirectory directory(directoryPath);
	if (!directory.exists()) {
		return "";
	}
	directory.listDir();
	directory.sort();
	for (const auto& file : directory.getFiles()) {
		if (!file.isFile()) {
			continue;
		}
		if (!isDiffusionModelExtension(file.getExtension())) {
			continue;
		}
		if (isLikelyAuxiliaryModel(file.getFileName())) {
			continue;
		}
		return file.getAbsolutePath();
	}
	return "";
}
}

void ofApp::setup() {
	ofSetWindowTitle("ofxGgmlDiffusion text-to-image example");
	gui.setup(nullptr, false);
	request = ofxGgmlDiffusionUtils::makeTextToImageRequest("a small generative texture study, openFrameworks, soft light");
	request.negativePrompt = "blurry, low quality";
	request.width = 512;
	request.height = 512;
	request.steps = 20;
	request.outputPath = getOutputPath();
	request.seed = -1;
	settings.modelPath = findModelPath();
	settings.photoMakerPath = findPhotoMakerPath();
	syncGuiFromRequest();
	writeBuffer(modelPathBuffer, settings.modelPath);
	loadPhotoMakerReferences();

	status = ofxGgmlDiffusionUtils::describe(request);
	if (!isNativeDiffusionBackendEnabled()) {
		detail = getNativeBackendSetupHint();
		ofLogWarning("ofxGgmlDiffusionPromptExample") << detail;
	} else if (settings.modelPath.empty()) {
		detail = "Set OFXGGML_DIFFUSION_MODEL or place a .safetensors, .gguf, or .ckpt model in bin/data/models";
	} else {
		detail = "Press R to generate with " + settings.modelPath;
	}
	ofLogNotice("ofxGgmlDiffusionPromptExample") << status;
	ofLogNotice("ofxGgmlDiffusionPromptExample") << detail;
}

void ofApp::keyPressed(int key) {
	if (key == 'r' || key == 'R') {
		runGeneration();
	} else if (key == 'c' || key == 'C') {
		runner.cancel();
		status = "cancelling";
		detail = "Waiting for stable-diffusion.cpp to return control";
	}
}

void ofApp::update() {
	ofxGgmlDiffusionResult result;
	if (runner.consumeResult(result)) {
		applyResult(result);
	} else if (runner.isRunning()) {
		status = runner.getStatus();
	}
}

void ofApp::runGeneration() {
	if (runner.isRunning()) {
		detail = "Generation is already running";
		return;
	}

	syncRequestFromGui();
	const auto validation = ofxGgmlDiffusionUtils::validate(request);
	if (!validation) {
		status = "request invalid";
		detail = validation.errors.empty() ? "Diffusion request failed validation" : validation.errors.front();
		ofLogWarning("ofxGgmlDiffusionPromptExample") << detail;
		return;
	}

	status = "starting";
	ofLogNotice("ofxGgmlDiffusionPromptExample") << ofxGgmlDiffusionUtils::describe(request);
	auto startResult = runner.start(settings, request);
	if (!startResult) {
		status = "start failed";
		detail = startResult.error;
		ofLogWarning("ofxGgmlDiffusionPromptExample") << detail;
		return;
	}
	detail = "Running on worker thread; press C to cancel the pending result";
}

void ofApp::applyResult(const ofxGgmlDiffusionResult& result) {
	if (!result) {
		status = ofxGgmlDiffusionGetTaskStateName(runner.getState());
		detail = result.error;
		ofLogWarning("ofxGgmlDiffusionPromptExample") << detail;
		return;
	}

	auto savedResult = result;
	if (!ofxGgmlDiffusionImageUtils::saveFirstImage(savedResult, request.outputPath)) {
		status = "save failed";
		detail = savedResult.error;
		ofLogWarning("ofxGgmlDiffusionPromptExample") << detail;
		return;
	}

	ofPixels pixels;
	if (ofxGgmlDiffusionImageUtils::toPixels(savedResult.images.front(), pixels)) {
		texture.loadData(pixels);
	}
	status = "complete";
	detail = "Saved " + savedResult.outputPath + " in " + ofToString(savedResult.elapsedMs, 0) + " ms";
	ofLogNotice("ofxGgmlDiffusionPromptExample") << detail;
}

void ofApp::syncRequestFromGui() {
	request.prompt = ofxGgmlDiffusionUtils::cleanPrompt(promptBuffer.data());
	request.negativePrompt = ofxGgmlDiffusionUtils::cleanPrompt(negativePromptBuffer.data());
	request.width = clampToMultipleOf64(request.width);
	request.height = clampToMultipleOf64(request.height);
	request.steps = std::max(1, request.steps);
	request.batchCount = std::clamp(request.batchCount, 1, 8);
	request.scheduler = indexToScheduler(schedulerIndex);
	request.seed = randomSeed ? -1 : std::max<std::int64_t>(1, seed);
	if (autoCfgScale) {
		request.cfgScale = std::numeric_limits<float>::infinity();
	} else {
		cfgScale = std::clamp(cfgScale, 0.0f, 30.0f);
		request.cfgScale = cfgScale;
	}
	settings.modelPath = modelPathBuffer.data();
	settings.threads = std::max(-1, settings.threads);
	request.outputPath = getOutputPath();
}

void ofApp::syncGuiFromRequest() {
	writeBuffer(promptBuffer, request.prompt);
	writeBuffer(negativePromptBuffer, request.negativePrompt);
	schedulerIndex = schedulerToIndex(request.scheduler);
	randomSeed = request.seed < 0;
	seed = request.seed < 0 ? 1 : static_cast<int>(request.seed);
	autoCfgScale = ofxGgmlDiffusionUtils::isAutoValue(request.cfgScale);
	if (!autoCfgScale) {
		cfgScale = request.cfgScale;
	}
}

std::string ofApp::findModelPath() const {
	const auto envModel = getEnvironmentVariable("OFXGGML_DIFFUSION_MODEL");
	if (!envModel.empty() && ofFile::doesFileExist(envModel, false)) {
		return envModel;
	}

	const std::vector<std::string> candidates = {
		ofToDataPath("models/model.safetensors", true),
		ofToDataPath("models/model.gguf", true),
		ofToDataPath("model.safetensors", true),
		ofToDataPath("model.gguf", true)
	};
	for (const auto& candidate : candidates) {
		if (ofFile::doesFileExist(candidate, false)) {
			return candidate;
		}
	}
	return findFirstDiffusionModelInDirectory(ofToDataPath("models", true));
}

std::string ofApp::findPhotoMakerPath() const {
	const auto envModel = getEnvironmentVariable("OFXGGML_PHOTOMAKER_MODEL");
	if (!envModel.empty() && ofFile::doesFileExist(envModel, false)) {
		return envModel;
	}
	const auto candidate = ofToDataPath("models/photomaker.safetensors", true);
	if (ofFile::doesFileExist(candidate, false)) {
		return candidate;
	}
	return "";
}

bool ofApp::loadPhotoMakerReferences() {
	if (settings.photoMakerPath.empty()) {
		return false;
	}
	const auto envRefs = getEnvironmentVariable("OFXGGML_PHOTOMAKER_REFS");
	if (envRefs.empty()) {
		return false;
	}

	std::vector<std::string> paths;
	std::stringstream stream(envRefs);
	std::string path;
	while (std::getline(stream, path, ';')) {
		path = ofTrim(path);
		if (!path.empty()) {
			paths.push_back(path);
		}
	}
	if (paths.empty()) {
		return false;
	}

	auto adapter = ofxGgmlDiffusionUtils::makePhotoMakerAdapter(
		settings.photoMakerPath,
		paths,
		"img");
	for (const auto& referencePath : paths) {
		ofxGgmlDiffusionImage image;
		if (!ofxGgmlDiffusionImageUtils::loadImage(referencePath, image)) {
			ofLogWarning("ofxGgmlDiffusionPromptExample") << "Skipping PhotoMaker reference: " << referencePath;
			continue;
		}
		adapter.referenceImages.push_back(std::move(image));
	}
	if (adapter.referenceImages.empty()) {
		ofLogWarning("ofxGgmlDiffusionPromptExample") << "PhotoMaker was configured but no reference images loaded";
		return false;
	}
	request.identityAdapter = std::move(adapter);
	request.modelFamily = ofxGgmlDiffusionModelFamily::SDXL;
	ofLogNotice("ofxGgmlDiffusionPromptExample") << "Loaded "
		<< request.identityAdapter.referenceImages.size()
		<< " PhotoMaker reference image(s)";
	return true;
}

std::string ofApp::getOutputPath() const {
	const auto outputDir = ofToDataPath("outputs", true);
	ofDirectory::createDirectory(outputDir, false, true);
	return ofFilePath::join(outputDir, "ofxGgmlDiffusionPromptExample.png");
}

void ofApp::draw() {
	ofBackground(18);
	gui.begin();
	ImGui::SetNextWindowPos(ImVec2(24.0f, 24.0f), ImGuiCond_Once);
	ImGui::SetNextWindowSize(ImVec2(660.0f, std::min(760.0f, ofGetHeight() - 48.0f)), ImGuiCond_Once);
	if (ImGui::Begin("ofxGgmlDiffusion Prompt Example")) {
		if (runner.isRunning()) {
			if (ImGui::Button("Cancel")) {
				runner.cancel();
				status = "cancelling";
				detail = "Waiting for stable-diffusion.cpp to return control";
			}
		} else if (ImGui::Button("Run")) {
			runGeneration();
		}
		ImGui::SameLine();
		ImGui::TextWrapped("%s", status.c_str());

		drawPromptControls();
		drawSettingsControls();
		drawModelControls();

		ImGui::TextUnformatted("PhotoMaker");
		ImGui::TextWrapped("%s", settings.photoMakerPath.empty() ? "(unset)" : settings.photoMakerPath.c_str());
		ImGui::Separator();
		ImGui::TextWrapped("%s", detail.c_str());
	}
	ImGui::End();
	gui.end();
	gui.draw();

	if (texture.isAllocated()) {
		const float maxSize = 512.0f;
		const float scale = std::min(maxSize / texture.getWidth(), maxSize / texture.getHeight());
		texture.draw(672, 32, texture.getWidth() * scale, texture.getHeight() * scale);
	}
}

void ofApp::drawPromptControls() {
	ImGui::Separator();
	ImGui::TextUnformatted("Prompt");
	if (ImGui::InputTextMultiline("##Prompt", promptBuffer.data(), promptBuffer.size(), ImVec2(-1.0f, 96.0f))) {
		syncRequestFromGui();
		status = ofxGgmlDiffusionUtils::describe(request);
	}
	ImGui::TextUnformatted("Negative prompt");
	if (ImGui::InputTextMultiline("##NegativePrompt", negativePromptBuffer.data(), negativePromptBuffer.size(), ImVec2(-1.0f, 72.0f))) {
		syncRequestFromGui();
	}
}

void ofApp::drawSettingsControls() {
	ImGui::Separator();
	bool changed = false;
	changed |= ImGui::SliderInt("Width", &request.width, 64, 2048);
	changed |= ImGui::SliderInt("Height", &request.height, 64, 2048);
	changed |= ImGui::SliderInt("Steps", &request.steps, 1, 100);
	changed |= ImGui::SliderInt("Batch", &request.batchCount, 1, 8);
	if (ImGui::Combo("Scheduler", &schedulerIndex, schedulerNames, 5)) {
		changed = true;
	}
	if (ImGui::Checkbox("Random seed", &randomSeed)) {
		changed = true;
	}
	if (!randomSeed) {
		changed |= ImGui::InputInt("Seed", &seed);
	}
	if (ImGui::Checkbox("Auto CFG", &autoCfgScale)) {
		changed = true;
	}
	if (!autoCfgScale) {
		changed |= ImGui::SliderFloat("CFG scale", &cfgScale, 0.0f, 30.0f, "%.2f");
	}
	changed |= ImGui::InputInt("Threads", &settings.threads);
	changed |= ImGui::Checkbox("Memory map", &settings.mmap);
	changed |= ImGui::Checkbox("VAE tiling", &settings.vaeTiling);
	changed |= ImGui::Checkbox("Flash attention", &settings.flashAttention);
	if (changed) {
		syncRequestFromGui();
		status = ofxGgmlDiffusionUtils::describe(request);
	}
}

void ofApp::drawModelControls() {
	ImGui::Separator();
	ImGui::TextUnformatted("Model");
	if (ImGui::InputText("##ModelPath", modelPathBuffer.data(), modelPathBuffer.size())) {
		syncRequestFromGui();
	}
	ImGui::SameLine();
	if (ImGui::Button("Browse")) {
		auto result = ofSystemLoadDialog("Load diffusion model", false);
		if (result.bSuccess) {
			settings.modelPath = result.getPath();
			writeBuffer(modelPathBuffer, settings.modelPath);
			syncRequestFromGui();
		}
	}
	ImGui::SameLine();
	if (ImGui::Button("Auto")) {
		settings.modelPath = findModelPath();
		writeBuffer(modelPathBuffer, settings.modelPath);
		syncRequestFromGui();
	}
	ImGui::TextWrapped("%s", settings.modelPath.empty() ? "(unset)" : settings.modelPath.c_str());
}
