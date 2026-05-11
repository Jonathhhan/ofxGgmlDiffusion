#include "ofApp.h"

#include <algorithm>
#include <cstdlib>
#include <vector>

void ofApp::setup() {
	ofSetWindowTitle("ofxGgmlDiffusion text-to-image example");
	request = ofxGgmlDiffusionUtils::makeTextToImageRequest("a small generative texture study, openFrameworks, soft light");
	request.negativePrompt = "blurry, low quality";
	request.width = 512;
	request.height = 512;
	request.steps = 20;
	request.outputPath = getOutputPath();
	settings.modelPath = findModelPath();

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
	const char* envModel = std::getenv("OFXGGML_DIFFUSION_MODEL");
	if (envModel && ofFile::doesFileExist(envModel, false)) {
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

std::string ofApp::getOutputPath() const {
	const auto outputDir = ofToDataPath("outputs", true);
	ofDirectory::createDirectory(outputDir, false, true);
	return ofFilePath::join(outputDir, "ofxGgmlDiffusionPromptExample.png");
}

void ofApp::draw() {
	ofBackground(18);
	ofSetColor(240);
	ofDrawBitmapString("ofxGgmlDiffusion text-to-image", 32, 48);
	ofDrawBitmapString("prompt: " + request.prompt, 32, 78);
	ofDrawBitmapString("model: " + (settings.modelPath.empty() ? "(unset)" : settings.modelPath), 32, 108);
	ofDrawBitmapString("status: " + status, 32, 138);
	ofDrawBitmapString(detail, 32, 168);
	ofDrawBitmapString("R run   C cancel", 32, 198);
	if (texture.isAllocated()) {
		const float maxSize = 512.0f;
		const float scale = std::min(maxSize / texture.getWidth(), maxSize / texture.getHeight());
		texture.draw(32, 232, texture.getWidth() * scale, texture.getHeight() * scale);
	}
}
