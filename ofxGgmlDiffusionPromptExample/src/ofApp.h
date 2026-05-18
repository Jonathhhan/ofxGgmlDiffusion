#pragma once

#include "ofMain.h"
#if __has_include("ofxGgmlDiffusion.h")
#include "ofxGgmlDiffusion.h"
#elif __has_include("../../src/ofxGgmlDiffusion.h")
#include "../../src/ofxGgmlDiffusion.h"
#else
#error "Cannot find ofxGgmlDiffusion.h. Ensure the ofxGgmlDiffusion addon is linked correctly."
#endif
#if __has_include("ofxImGui.h")
#include "ofxImGui.h"
#elif __has_include("../../../ofxImGui/src/ofxImGui.h")
#include "../../../ofxImGui/src/ofxImGui.h"
#else
#error "Cannot find ofxImGui.h. Ensure the ofxImGui addon is linked correctly."
#endif

#include <array>
#include <cstdint>
#include <string>

class ofApp : public ofBaseApp {
public:
	void setup() override;
	void update() override;
	void keyPressed(int key) override;
	void draw() override;

private:
	void runGeneration();
	void drawPromptControls();
	void drawSettingsControls();
	void drawModelControls();
	void syncRequestFromGui();
	void syncGuiFromRequest();
	std::string findModelPath() const;
	std::string findPhotoMakerPath() const;
	bool loadPhotoMakerReferences();
	std::string getOutputPath() const;
	void applyResult(const ofxGgmlDiffusionResult& result);

	ofxGgmlDiffusionAsyncRunner runner;
	ofxGgmlDiffusionContextSettings settings;
	ofxGgmlDiffusionRequest request;
	ofxImGui::Gui gui;
	ofTexture texture;
	std::array<char, 2048> promptBuffer{};
	std::array<char, 2048> negativePromptBuffer{};
	std::array<char, 1024> modelPathBuffer{};
	int schedulerIndex = 0;
	int seed = -1;
	float cfgScale = 7.0f;
	bool autoCfgScale = true;
	bool randomSeed = true;
	std::string status;
	std::string detail;
};
