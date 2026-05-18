param(
	[string]$Configuration = "Release",
	[string]$Platform = "x64",
	[string]$Example = "ofxGgmlDiffusionPromptExample",
	[int]$Jobs = 1,
	[switch]$Clean
)

$ErrorActionPreference = "Stop"

function Write-Step {
	param([string]$Message)
	Write-Host "==> $Message"
}

function Test-WindowsHost {
	return !($IsLinux -or $IsMacOS)
}

function Invoke-CheckedNative {
	param(
		[string]$Step,
		[scriptblock]$Command
	)
	& $Command
	if ($LASTEXITCODE -ne 0) {
		throw "$Step failed with exit code $LASTEXITCODE"
	}
}

function Get-StableNameFragment {
	param([string]$Text)
	$sha1 = [System.Security.Cryptography.SHA1]::Create()
	try {
		$bytes = [System.Text.Encoding]::UTF8.GetBytes($Text)
		$hash = $sha1.ComputeHash($bytes)
		return [System.BitConverter]::ToString($hash).Replace("-", "")
	} finally {
		$sha1.Dispose()
	}
}

function Invoke-WithNamedMutex {
	param(
		[string]$Name,
		[scriptblock]$Command
	)
	$mutex = New-Object System.Threading.Mutex($false, $Name)
	$locked = $false
	try {
		$locked = $mutex.WaitOne([TimeSpan]::FromMinutes(30))
		if (!$locked) {
			throw "Timed out waiting for build lock: $Name"
		}
		& $Command
	} finally {
		if ($locked) {
			$mutex.ReleaseMutex()
		}
		$mutex.Dispose()
	}
}

function Get-MsBuild {
	$vswhere = Join-Path ${env:ProgramFiles(x86)} "Microsoft Visual Studio\Installer\vswhere.exe"
	if (Test-Path -LiteralPath $vswhere) {
		$installPath = & $vswhere -latest -products * -requires Microsoft.Component.MSBuild -property installationPath 2>$null
		if ($installPath) {
			$candidate = Join-Path $installPath "MSBuild\Current\Bin\MSBuild.exe"
			if (Test-Path -LiteralPath $candidate) {
				return $candidate
			}
		}
	}

	foreach ($version in @("18", "17", "16")) {
		foreach ($edition in @("Community", "Professional", "Enterprise", "BuildTools")) {
			$candidate = "C:\Program Files\Microsoft Visual Studio\$version\$edition\MSBuild\Current\Bin\MSBuild.exe"
			if (Test-Path -LiteralPath $candidate) {
				return $candidate
			}
		}
	}
	return ""
}

function Resolve-BuildJobs {
	param([int]$RequestedJobs)
	if ($RequestedJobs -lt 0) {
		throw "-Jobs must be 0 or greater."
	}
	if ($RequestedJobs -eq 0) {
		return [Environment]::ProcessorCount
	}
	return $RequestedJobs
}

function Get-MsBuildParallelArguments {
	param([int]$BuildJobs)
	if ($BuildJobs -gt 1) {
		return @("/p:MultiProcessorCompilation=true", "/m:$BuildJobs")
	}
	return @("/p:MultiProcessorCompilation=false", "/m:1")
}

function Get-RelativeProjectPath {
	param(
		[string]$ProjectDir,
		[string]$FilePath
	)
	$projectUri = [System.Uri]((Resolve-Path -LiteralPath $ProjectDir).Path.TrimEnd("\") + "\")
	$fileUri = [System.Uri](Resolve-Path -LiteralPath $FilePath).Path
	return [System.Uri]::UnescapeDataString(
		$projectUri.MakeRelativeUri($fileUri).ToString()).Replace("/", "\")
}

function Get-ExampleAddons {
	param([string]$ExampleRoot)
	$addonsMake = Join-Path $ExampleRoot "addons.make"
	if (!(Test-Path -LiteralPath $addonsMake)) {
		return @()
	}
	return Get-Content -LiteralPath $addonsMake |
		ForEach-Object { $_.Trim() } |
		Where-Object { $_ -and -not $_.StartsWith("#") }
}

function Get-AddonSourceExcludes {
	param([string]$AddonRoot)
	$configValues = Get-AddonConfigValues -AddonRoot $AddonRoot
	if (!$configValues) {
		return @()
	}
	$excludePaths = New-Object System.Collections.Generic.List[string]
	foreach ($path in @($configValues["ADDON_SOURCES_EXCLUDE"])) {
		$normPath = ([string]$path).Trim() -replace '/', '\' -replace '%', '*'
		if (-not [string]::IsNullOrWhiteSpace($normPath) -and -not $excludePaths.Contains($normPath)) {
			$excludePaths.Add($normPath)
		}
	}
	foreach ($path in @($configValues["ADDON_INCLUDES_EXCLUDE"])) {
		$normPath = ([string]$path).Trim() -replace '/', '\' -replace '%', '*'
		if (-not [string]::IsNullOrWhiteSpace($normPath) -and -not $excludePaths.Contains($normPath)) {
			$excludePaths.Add($normPath)
		}
	}
	return @($excludePaths)
}

function Get-AddonConfigValues {
	param([string]$AddonRoot)

	$values = @{}
	$excludePaths = New-Object System.Collections.Generic.List[string]
	$configPath = Join-Path $AddonRoot "addon_config.mk"
	if (!(Test-Path -LiteralPath $configPath)) {
		return $values
	}
	$section = ""
	Get-Content -LiteralPath $configPath | ForEach-Object {
		$line = ([string]$_ -replace "\s+#.*$", "").Trim()
		if ([string]::IsNullOrWhiteSpace($line)) {
			return
		}
		if ($line -match '^([A-Za-z0-9_/]+):\s*$') {
			$section = $matches[1]
			return
		}
		if (($section -ne "common" -and $section -ne "vs")) {
			return
		}
		if ($line -match '^(ADDON_[A-Z_]+)\s*(?:\+)?=\s*(.+)$') {
			$name = $matches[1]
			foreach ($part in @($matches[2] -split '\s+' | Where-Object { -not [string]::IsNullOrWhiteSpace($_) })) {
				if (!$values.ContainsKey($name)) {
					$values[$name] = New-Object System.Collections.Generic.List[string]
				}
				if (!$values[$name].Contains($part)) {
					$values[$name].Add($part)
				}
			}
		}
	}
	return $values
}

function Test-AddonSourceExcluded {
	param(
		[string]$RelativePath,
		[string[]]$ExcludePatterns
	)
	$normalized = $RelativePath -replace '/', '\'
	foreach ($pattern in $ExcludePatterns) {
		if ($normalized -like $pattern) {
			return $true
		}
	}
	return $false
}

function Get-AddonSourceRoots {
	param([string]$AddonRoot)
	$roots = New-Object System.Collections.Generic.List[string]
	foreach ($name in @("src", "libs")) {
		$candidate = Join-Path $AddonRoot $name
		if (Test-Path -LiteralPath $candidate) {
			$roots.Add($candidate)
		}
	}
	return @($roots)
}

function Get-AddonIncludes {
	param([string]$AddonRoot)
	$includePaths = New-Object System.Collections.Generic.List[string]
	$includeFromMk = New-Object System.Collections.Generic.List[string]
	$excludePaths = Get-AddonSourceExcludes -AddonRoot $AddonRoot
	$config = Get-AddonConfigValues -AddonRoot $AddonRoot
	foreach ($path in @($config["ADDON_INCLUDES"])) {
		$tempPath = ([string]$path).Trim()
		if (-not [string]::IsNullOrWhiteSpace($tempPath)) {
			$normPath = $tempPath -replace '/', '\' -replace '%', '*'
			if (-not $includeFromMk.Contains($normPath)) {
				$includeFromMk.Add($normPath)
			}
		}
	}

	foreach ($path in $includeFromMk) {
		if (Test-AddonSourceExcluded -RelativePath $path -ExcludePatterns $excludePaths) {
			continue
		}
		$fullPath = Join-Path $AddonRoot $path
		if (Test-Path -LiteralPath $fullPath -PathType Container) {
			$includePaths.Add($fullPath)
		}
	}

	$srcRoot = Join-Path $AddonRoot "src"
	if (Test-Path -LiteralPath $srcRoot -PathType Container) {
		$includePaths.Add($srcRoot)
	}
	$libsRoot = Join-Path $AddonRoot "libs"
	if (Test-Path -LiteralPath $libsRoot -PathType Container) {
		Get-ChildItem -LiteralPath $libsRoot -Directory | ForEach-Object {
			$candidate = $_
			$relative = Get-RelativeProjectPath -ProjectDir $AddonRoot -FilePath $candidate.FullName
			if (Test-AddonSourceExcluded -RelativePath $relative -ExcludePatterns $excludePaths) {
				return
			}
			$includePaths.Add($candidate.FullName)
			foreach ($subName in @("src", "include", "backends")) {
				$subDir = Join-Path $candidate.FullName $subName
				if (Test-Path -LiteralPath $subDir -PathType Container) {
					$includePaths.Add($subDir)
				}
			}
			$extrasDir = Join-Path $candidate.FullName "extras"
			if (Test-Path -LiteralPath $extrasDir -PathType Container) {
				$includePaths.Add($extrasDir)
			}
		}
	}
	$hash = New-Object System.Collections.Generic.HashSet[string]
	foreach ($candidate in $includePaths) {
		$normalized = [System.IO.Path]::GetFullPath($candidate)
		$hash.Add($normalized) | Out-Null
	}
	return @($hash)
}

function Add-VisualStudioProjectItem {
	param(
		[xml]$Doc,
		[System.Xml.XmlNamespaceManager]$Namespace,
		[string]$Tag,
		[string]$Include
	)
	$existing = $Doc.SelectSingleNode("//msb:$Tag[@Include='$Include']", $Namespace)
	if ($existing) {
		return $false
	}
	$itemGroups = @($Doc.SelectNodes("//msb:ItemGroup", $Namespace))
	$itemGroup = $null
	foreach ($group in $itemGroups) {
		if ($group.SelectSingleNode("msb:$Tag", $Namespace)) {
			$itemGroup = $group
			break
		}
	}
	if (!$itemGroup -and $itemGroups.Count -gt 0) {
		$itemGroup = $itemGroups[0]
	}
	if (!$itemGroup) {
		return $false
	}
	$item = $Doc.CreateElement($Tag, $Doc.DocumentElement.NamespaceURI)
	$item.SetAttribute("Include", $Include)
	[void]$itemGroup.AppendChild($item)
	return $true
}

function Add-SemicolonNodeValue {
	param(
		[xml]$Doc,
		[System.Xml.XmlNamespaceManager]$Namespace,
		[string]$NodeName,
		[string]$Value,
		[switch]$Apply
	)

	if ([string]::IsNullOrWhiteSpace($Value)) {
		return $false
	}
	$nodes = @($Doc.SelectNodes("//msb:$NodeName", $Namespace))
	$changed = $false
	foreach ($node in $nodes) {
		$parts = New-Object System.Collections.Generic.List[string]
		foreach ($part in @($node.InnerText -split ";" | Where-Object { -not [string]::IsNullOrWhiteSpace($_) })) {
			$parts.Add([string]$part)
		}
		if (!$parts.Contains($Value)) {
			$changed = $true
			if ($Apply) {
				$parts.Add($Value)
				$node.InnerText = ($parts.ToArray() -join ";")
			}
		}
	}
	return $changed
}

function Add-SpaceNodeValue {
	param(
		[xml]$Doc,
		[System.Xml.XmlNamespaceManager]$Namespace,
		[string]$XPath,
		[string]$Value,
		[switch]$Apply
	)

	if ([string]::IsNullOrWhiteSpace($Value)) {
		return $false
	}
	$nodes = @($Doc.SelectNodes($XPath, $Namespace))
	$changed = $false
	foreach ($node in $nodes) {
		$parts = New-Object System.Collections.Generic.List[string]
		foreach ($part in @($node.InnerText -split "\s+" | Where-Object { -not [string]::IsNullOrWhiteSpace($_) })) {
			$parts.Add([string]$part)
		}
		if (!$parts.Contains($Value)) {
			$changed = $true
			if ($Apply) {
				$parts.Add($Value)
				$node.InnerText = ($parts.ToArray() -join " ")
			}
		}
	}
	return $changed
}

function Get-AddonCFlags {
	param([string]$AddonRoot)

	$flags = New-Object System.Collections.Generic.List[string]
	$config = Get-AddonConfigValues -AddonRoot $AddonRoot
	foreach ($flag in @($config["ADDON_CFLAGS"])) {
		$flags.Add([string]$flag)
	}
	return @($flags | Where-Object { $_ -notmatch '^\s*$' })
}

function Get-AddonLibs {
	param([string]$AddonRoot)

	$libraryReferences = New-Object System.Collections.Generic.List[string]
	$config = Get-AddonConfigValues -AddonRoot $AddonRoot
	foreach ($library in @($config["ADDON_LIBS"])) {
		$libraryReferences.Add([string]$library)
	}
	return @($libraryReferences | Where-Object { $_ -notmatch '^\s*$' })
}

function ConvertTo-ProjectLibraryReference {
	param(
		[string]$AddonRoot,
		[string]$ProjectDir,
		[string]$Library
	)

	$value = ([string]$Library).Trim().Trim('"')
	if ([string]::IsNullOrWhiteSpace($value)) {
		return $null
	}

	$normalized = $value -replace "/", "\"
	$parent = Split-Path -Parent $normalized
	$name = [System.IO.Path]::GetFileName($normalized)
	$directory = ""
	if ($parent) {
		if ($parent -match '^\$\(' -or [System.IO.Path]::IsPathRooted($parent)) {
			$directory = $parent
		} else {
			$libraryPath = Join-Path $AddonRoot $parent
			if (Test-Path -LiteralPath $libraryPath -PathType Container) {
				$directory = Get-RelativeProjectPath -ProjectDir $ProjectDir -FilePath $libraryPath
			}
		}
	}

	return [pscustomobject]@{
		Dependency = $name
		Directory = $directory
	}
}

function Remove-StaleDependencyProjectItems {
	param(
		[xml]$Doc,
		[System.Xml.XmlNamespaceManager]$Namespace,
		[string[]]$AllowedClCompile,
		[string[]]$AllowedClInclude,
		[string[]]$ManagedPrefixes
	)
	$managedPrefixes = @($ManagedPrefixes)
	$allowedCompile = New-Object "System.Collections.Generic.HashSet[string]" ([StringComparer]::OrdinalIgnoreCase)
	$allowedInclude = New-Object "System.Collections.Generic.HashSet[string]" ([StringComparer]::OrdinalIgnoreCase)
	foreach ($item in $AllowedClCompile) { [void]$allowedCompile.Add($item) }
	foreach ($item in $AllowedClInclude) { [void]$allowedInclude.Add($item) }

	$changed = $false
	foreach ($group in @($Doc.SelectNodes("//msb:ItemGroup", $Namespace))) {
		foreach ($node in @($group.SelectNodes("msb:ClCompile|msb:ClInclude", $Namespace))) {
			$include = $node.Include
			if ([string]::IsNullOrWhiteSpace($include)) {
				continue
			}
			$isManaged = $false
			foreach ($prefix in $managedPrefixes) {
				if ($include -like "${prefix}*") {
					$isManaged = $true
					break
				}
			}
			if (-not $isManaged) {
				continue
			}
			if ($node.LocalName -eq "ClCompile") {
				if (-not $allowedCompile.Contains($include)) {
					$group.RemoveChild($node) | Out-Null
					$changed = $true
				}
			} elseif ($node.LocalName -eq "ClInclude") {
				if (-not $allowedInclude.Contains($include)) {
					$group.RemoveChild($node) | Out-Null
					$changed = $true
				}
			}
		}
	}
	return $changed
}

function Repair-DiffusionGeneratedProject {
	param(
		[string]$Project,
		[string]$AddonRoot
	)
	if (!(Test-Path -LiteralPath $Project -PathType Leaf)) {
		return
	}

	[xml]$doc = Get-Content -LiteralPath $Project -Raw
	$namespace = New-Object System.Xml.XmlNamespaceManager($doc.NameTable)
	$namespace.AddNamespace("msb", "http://schemas.microsoft.com/developer/msbuild/2003")
	$projectDir = Split-Path -Parent $Project
	$changed = $false

	$addonsDir = Split-Path -Path $AddonRoot -Parent
	$addons = Get-ExampleAddons -ExampleRoot $projectDir
	if ($addons.Count -eq 0) {
		$addons = @("ofxGgmlDiffusion")
	}
	if ($addons -notcontains "ofxGgmlDiffusion") {
		$addons = @("ofxGgmlDiffusion") + $addons
	}
	$includeDirs = New-Object System.Collections.Generic.HashSet[string]
	$compilerFlags = New-Object System.Collections.Generic.HashSet[string] ([StringComparer]::OrdinalIgnoreCase)
	$dependencyReferences = New-Object System.Collections.Generic.HashSet[string] ([StringComparer]::OrdinalIgnoreCase)
	$dependencyDirectories = New-Object System.Collections.Generic.HashSet[string] ([StringComparer]::OrdinalIgnoreCase)
	foreach ($addon in $addons) {
		$addonRoot = Join-Path $addonsDir $addon
		if (!(Test-Path -LiteralPath $addonRoot)) {
			continue
		}
		foreach ($absoluteInclude in Get-AddonIncludes -AddonRoot $addonRoot) {
			$relativeInclude = Get-RelativeProjectPath -ProjectDir $projectDir -FilePath $absoluteInclude
			$normalizedRelativeInclude = $relativeInclude -replace '/', '\'
			if (-not $includeDirs.Contains($normalizedRelativeInclude)) {
				$includeDirs.Add($normalizedRelativeInclude) | Out-Null
			}
		}
		foreach ($flag in Get-AddonCFlags -AddonRoot $addonRoot) {
			if (![string]::IsNullOrWhiteSpace($flag)) {
				$compilerFlags.Add($flag) | Out-Null
			}
		}
		foreach ($library in Get-AddonLibs -AddonRoot $addonRoot) {
			$reference = ConvertTo-ProjectLibraryReference -AddonRoot $addonRoot -ProjectDir $projectDir -Library ([string]$library)
			if (!$reference) {
				continue
			}
			if ($reference.Dependency -and -not $dependencyReferences.Contains($reference.Dependency)) {
				$dependencyReferences.Add($reference.Dependency) | Out-Null
			}
			if ($reference.Directory -and -not $dependencyDirectories.Contains($reference.Directory)) {
				$dependencyDirectories.Add($reference.Directory) | Out-Null
			}
		}
	}
	foreach ($node in @($doc.SelectNodes("//msb:AdditionalIncludeDirectories", $namespace))) {
		$parts = @($node.InnerText -split ";" | Where-Object { $_ })
		foreach ($includeDir in $includeDirs) {
			if ($parts -notcontains $includeDir) {
				$parts += $includeDir
				$changed = $true
			}
		}
		$node.InnerText = $parts -join ";"
	}

	foreach ($flag in @($compilerFlags | Sort-Object)) {
		if (Add-SpaceNodeValue -Doc $doc -Namespace $namespace -XPath "//msb:ClCompile/msb:AdditionalOptions" -Value $flag -Apply) {
			$changed = $true
		}
	}

	if ($dependencyReferences.Count -gt 0) {
		foreach ($dependency in @($dependencyReferences | Sort-Object)) {
			if (Add-SemicolonNodeValue -Doc $doc -Namespace $namespace -NodeName "AdditionalDependencies" -Value $dependency -Apply) {
				$changed = $true
			}
		}
		foreach ($directory in @($dependencyDirectories | Sort-Object)) {
			if (Add-SemicolonNodeValue -Doc $doc -Namespace $namespace -NodeName "AdditionalLibraryDirectories" -Value $directory -Apply) {
				$changed = $true
			}
		}
	}

	$desiredClCompile = New-Object "System.Collections.Generic.HashSet[string]" ([StringComparer]::OrdinalIgnoreCase)
	$desiredClInclude = New-Object "System.Collections.Generic.HashSet[string]" ([StringComparer]::OrdinalIgnoreCase)
	$managedPrefixes = @()

	foreach ($addon in $addons) {
		$addonRoot = Join-Path $addonsDir $addon
		if (!(Test-Path -LiteralPath $addonRoot)) {
			Write-Step "Skipping missing addon path for $addon"
			continue
		}
		$managedPrefixes += "..\..\" + $addon + "\"
		$excludePaths = Get-AddonSourceExcludes -AddonRoot $addonRoot
		foreach ($sourceRoot in Get-AddonSourceRoots -AddonRoot $addonRoot) {
			Get-ChildItem -LiteralPath $sourceRoot -Recurse -File | ForEach-Object {
				$relativeToAddon = Get-RelativeProjectPath -ProjectDir $addonRoot -FilePath $_.FullName
				$normalizedRelative = $relativeToAddon -replace '/', '\'
				if (-not (Test-AddonSourceExcluded -RelativePath $normalizedRelative -ExcludePatterns $excludePaths)) {
					$relative = Get-RelativeProjectPath -ProjectDir $projectDir -FilePath $_.FullName
					if ($_.Extension -in @(".cpp", ".cxx", ".cc")) {
						[void]$desiredClCompile.Add($relative)
						if (Add-VisualStudioProjectItem -Doc $doc -Namespace $namespace -Tag "ClCompile" -Include $relative) {
							$changed = $true
						}
					} elseif ($_.Extension -in @(".h", ".hpp")) {
						[void]$desiredClInclude.Add($relative)
						if (Add-VisualStudioProjectItem -Doc $doc -Namespace $namespace -Tag "ClInclude" -Include $relative) {
							$changed = $true
						}
					}
				}
			}
		}
	}

	$managedPrefixes = @($managedPrefixes | Select-Object -Unique)
	if (Remove-StaleDependencyProjectItems -Doc $doc -Namespace $namespace -AllowedClCompile $desiredClCompile -AllowedClInclude $desiredClInclude -ManagedPrefixes $managedPrefixes) {
		$changed = $true
	}

	foreach ($node in @($doc.SelectNodes("//msb:PostBuildEvent/msb:Command", $namespace))) {
		if ($node.InnerText -match '\$\(ProjectDir\)dll\\([^\\]+)\\\*\.dll') {
			$platformName = $matches[1]
			$guardedCommand = "if exist `"`$(ProjectDir)dll\$platformName\*.dll`" xcopy /Y /E `"`$(ProjectDir)dll\$platformName\*.dll`" `"`$(TargetDir)`""
			if ($node.InnerText -ne $guardedCommand) {
				$node.InnerText = $guardedCommand
				$changed = $true
			}
		}
	}

	if ($changed) {
		$doc.Save($Project)
		Write-Step "Repaired generated Visual Studio project metadata"
	}
}

function Initialize-DiffusionGeneratedProject {
	param(
		[string]$ExampleRoot,
		[string]$Example,
		[string]$OfRoot
	)
	$templateRoot = Join-Path $OfRoot "scripts\templates\winvs"
	$templateProject = Join-Path $templateRoot "emptyExample.vcxproj"
	if (!(Test-Path -LiteralPath $templateProject -PathType Leaf)) {
		throw "Visual Studio project not found and openFrameworks winvs template is missing: $templateProject"
	}
	$copies = @(
		@{ Source = "emptyExample.vcxproj"; Target = "$Example.vcxproj" },
		@{ Source = "emptyExample.vcxproj.filters"; Target = "$Example.vcxproj.filters" },
		@{ Source = "emptyExample.vcxproj.user"; Target = "$Example.vcxproj.user" },
		@{ Source = "emptyExample.sln"; Target = "$Example.sln" },
		@{ Source = "icon.rc"; Target = "icon.rc" }
	)
	foreach ($copy in $copies) {
		$source = Join-Path $templateRoot $copy.Source
		$target = Join-Path $ExampleRoot $copy.Target
		if (!(Test-Path -LiteralPath $source -PathType Leaf)) {
			continue
		}
		$text = Get-Content -LiteralPath $source -Raw
		$text = $text.Replace("emptyExample", $Example)
		Set-Content -LiteralPath $target -Value $text -NoNewline
	}
	Write-Step "Initialized generated Visual Studio project metadata from openFrameworks template"
}

$scriptRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$addonRoot = Resolve-Path (Join-Path $scriptRoot "..")
$ofRoot = Resolve-Path (Join-Path $addonRoot "..\..")
$exampleRoot = Join-Path $addonRoot $Example
if (!(Test-Path -LiteralPath $exampleRoot -PathType Container)) {
	throw "Example directory not found: $exampleRoot"
}

if (Test-WindowsHost) {
	$project = Join-Path $exampleRoot "$Example.vcxproj"
	if (!(Test-Path -LiteralPath $project -PathType Leaf)) {
		Initialize-DiffusionGeneratedProject -ExampleRoot $exampleRoot -Example $Example -OfRoot $ofRoot
	}
	Repair-DiffusionGeneratedProject -Project $project -AddonRoot $addonRoot
	$msbuild = Get-MsBuild
	if ([string]::IsNullOrWhiteSpace($msbuild)) {
		throw "MSBuild.exe was not found."
	}

	$target = if ($Clean) { "Rebuild" } else { "Build" }
	$buildJobs = Resolve-BuildJobs -RequestedJobs $Jobs
	$parallelArgs = Get-MsBuildParallelArguments -BuildJobs $buildJobs
	Write-Step "Building $Example $Configuration $Platform with MSBuild ($buildJobs jobs)"
	$lockName = "Local\ofxGgml-msbuild-" + (Get-StableNameFragment $ofRoot.Path)
	Invoke-WithNamedMutex -Name $lockName -Command {
		& $msbuild $project /t:$target /p:Configuration=$Configuration /p:Platform=$Platform @parallelArgs /nr:false
		if ($LASTEXITCODE -ne 0) {
			throw "MSBuild $Example failed with exit code $LASTEXITCODE"
		}
	}
	return
}

$makefile = Join-Path $exampleRoot "Makefile"
if (Test-Path -LiteralPath $makefile -PathType Leaf) {
	$target = if ($Clean) { "clean Release" } else { "Release" }
	Write-Step "Building $Example with make"
	Invoke-CheckedNative "make $Example" {
		make -C $exampleRoot $target
	}
	return
}

if ($IsMacOS) {
	$xcodeProject = Get-ChildItem -LiteralPath $exampleRoot -Filter "*.xcodeproj" -Directory -ErrorAction SilentlyContinue | Select-Object -First 1
	if ($xcodeProject) {
		Write-Step "Building $Example $Configuration with xcodebuild"
		Invoke-CheckedNative "xcodebuild $Example" {
			xcodebuild -project $xcodeProject.FullName -configuration $Configuration
		}
		return
	}
}

throw "No supported generated project was found for $Example. Generate the example project with openFrameworks projectGenerator first."
