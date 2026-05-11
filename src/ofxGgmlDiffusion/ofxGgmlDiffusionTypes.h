#pragma once

#include <string>
#include <vector>

struct ofxGgmlDiffusionRequest {
	std::string prompt;
	std::string negativePrompt;
	std::vector<std::string> tags;
};

struct ofxGgmlDiffusionResult {
	bool success = false;
	std::string text;
	std::string error;
	std::vector<std::string> references;

	explicit operator bool() const {
		return success;
	}
};