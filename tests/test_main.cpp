#include "ofxGgmlDiffusion.h"

#include <iostream>

int main() {
	ofxGgmlDiffusionRequest request;
	if (ofxGgmlDiffusionUtils::hasInput(request)) {
		std::cerr << "empty request reported as configured\n";
		return 1;
	}

	request.prompt = "a small generative texture study";
	if (!ofxGgmlDiffusionUtils::hasInput(request)) {
		std::cerr << "configured request reported as empty\n";
		return 1;
	}
	request.width = 512;
	request.height = 512;

	const auto description = ofxGgmlDiffusionUtils::describe(request);
	if (description.find(request.prompt) == std::string::npos) {
		std::cerr << "description did not include request input\n";
		return 1;
	}
	if (description.find("512x512") == std::string::npos) {
		std::cerr << "description did not include dimensions\n";
		return 1;
	}

	const auto valid = ofxGgmlDiffusionUtils::validate(request);
	if (!valid) {
		std::cerr << "valid request failed validation\n";
		return 1;
	}

	auto invalid = ofxGgmlDiffusionUtils::makeTextToImageRequest("  a   cat  ");
	invalid.width = 513;
	const auto invalidResult = ofxGgmlDiffusionUtils::validate(invalid);
	if (invalidResult) {
		std::cerr << "invalid dimension passed validation\n";
		return 1;
	}
	if (invalid.prompt != "a cat") {
		std::cerr << "prompt was not cleaned\n";
		return 1;
	}

	auto inpaint = request;
	inpaint.mode = ofxGgmlDiffusionMode::Inpainting;
	if (ofxGgmlDiffusionUtils::validate(inpaint)) {
		std::cerr << "inpainting without images passed validation\n";
		return 1;
	}

	return 0;
}
