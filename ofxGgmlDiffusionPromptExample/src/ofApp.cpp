#include "ofApp.h"

#include "imgui.h"

#include <algorithm>
#include <cstdlib>
#include <sstream>
#include <utility>
#include <vector>

namespace {
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
	settings.modelPath = findModelPath();
	settings.photoMakerPath = findPhotoMakerPath();
	loadPhotoMakerReferences();

	status = ofxGgmlDiffusionUtils::describe(request);
	if (settings.modelPath.empty()) {
		detail = "Set OFXGGML_DIFFUSION_MODEL or place a model at bin/data/models/model.safetensors";
	} else {
		detail = "Press R to generate";
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
	return "";
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
	ImGui::SetNextWindowSize(ImVec2(620.0f, 360.0f), ImGuiCond_Once);
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

		ImGui::Separator();
		ImGui::TextUnformatted("Prompt");
		ImGui::TextWrapped("%s", request.prompt.c_str());
		ImGui::TextUnformatted("Model");
		ImGui::TextWrapped("%s", settings.modelPath.empty() ? "(unset)" : settings.modelPath.c_str());
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
