#pragma once

#include "ofMain.h"
#include "ofxGgmlDiffusion.h"

class ofApp : public ofBaseApp {
public:
	void setup() override;
	void draw() override;

private:
	ofxGgmlDiffusionRequest request;
	std::string status;
};