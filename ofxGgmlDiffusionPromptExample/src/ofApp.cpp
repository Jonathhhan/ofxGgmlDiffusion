#include "ofApp.h"

void ofApp::setup() {
	ofSetWindowTitle("ofxGgmlDiffusion smoke example");
	request.prompt = "a small generative texture study";
	status = ofxGgmlDiffusionUtils::describe(request);
	ofLogNotice("ofxGgmlDiffusionPromptExample") << status;
}

void ofApp::draw() {
	ofBackground(18);
	ofSetColor(240);
	ofDrawBitmapString("ofxGgmlDiffusion", 32, 48);
	ofDrawBitmapString(status, 32, 78);
}