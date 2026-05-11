#include "ofApp.h"

#include <cstdlib>
#include <vector>

void ofApp::setup() {
	ofSetWindowTitle("ofxGgmlDiffusion GAN example");
	prompt = "small monochrome icon set for an openFrameworks tool";
	backend = ofxGgmlMakeUnavailableDiffusionImageGenerationBackend(
		ofxGgmlDiffusionBackendFamily::GAN,
		"gan");
	rebuildRequest();

	status = ofxGgmlDiffusionUtils::describe(request);
	if (request.gan.generatorPath.empty()) {
		detail = "Set OFXGGML_GAN_GENERATOR or place a generator at bin/data/models/generator.gguf";
	} else {
		detail = "Press R to run the GAN request";
	}
	ofLogNotice("ofxGgmlDiffusionGanExample") << status;
	ofLogNotice("ofxGgmlDiffusionGanExample") << detail;
}

void ofApp::keyPressed(int key) {
	if (key == 'r' || key == 'R') {
		runGeneration();
	}
}

void ofApp::runGeneration() {
	rebuildRequest();
	const auto validation = ofxGgmlDiffusionUtils::validate(request);
	if (!validation) {
		status = "request invalid";
		detail = validation.errors.empty() ? "GAN request failed validation" : validation.errors.front();
		ofLogWarning("ofxGgmlDiffusionGanExample") << detail;
		return;
	}

	status = "starting";
	ofLogNotice("ofxGgmlDiffusionGanExample") << ofxGgmlDiffusionUtils::describe(request);
	auto setupResult = backend->setup(settings);
	if (!setupResult) {
		status = "backend unavailable";
		detail = setupResult.error;
		ofLogWarning("ofxGgmlDiffusionGanExample") << detail;
		return;
	}

	auto result = backend->generate(request);
	if (!result) {
		status = "generation failed";
		detail = result.error;
		ofLogWarning("ofxGgmlDiffusionGanExample") << detail;
		return;
	}

	status = "complete";
	detail = result.outputPath.empty() ? "GAN generation completed" : "Saved " + result.outputPath;
	ofLogNotice("ofxGgmlDiffusionGanExample") << detail;
}

void ofApp::rebuildRequest() {
	request = ofxGgmlDiffusionUtils::makeGanImageRequest(prompt, findGeneratorPath());
	request.width = 512;
	request.height = 512;
	request.outputPath = getOutputPath();
	request.seed = 1234;
	request.gan.latentSize = 512;
	request.gan.truncation = 0.85f;
}

std::string ofApp::findGeneratorPath() const {
	const char* envGenerator = std::getenv("OFXGGML_GAN_GENERATOR");
	if (envGenerator && ofFile::doesFileExist(envGenerator, false)) {
		return envGenerator;
	}

	const std::vector<std::string> candidates = {
		ofToDataPath("models/generator.gguf", true),
		ofToDataPath("models/gan-generator.gguf", true),
		ofToDataPath("models/generator.bin", true),
		ofToDataPath("generator.gguf", true)
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
	return ofFilePath::join(outputDir, "ofxGgmlDiffusionGanExample.png");
}

void ofApp::draw() {
	ofBackground(18);
	ofSetColor(240);
	ofDrawBitmapString("ofxGgmlDiffusion GAN image generation", 32, 48);
	ofDrawBitmapString("prompt: " + prompt, 32, 78);
	ofDrawBitmapString("backend: " + backend->getBackendName(), 32, 108);
	ofDrawBitmapString("generator: " + (request.gan.generatorPath.empty() ? "(unset)" : request.gan.generatorPath), 32, 138);
	ofDrawBitmapString("status: " + status, 32, 168);
	ofDrawBitmapString(detail, 32, 198);
	ofDrawBitmapString("R run request", 32, 228);
}
