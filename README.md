# CUDA Real Time Path Tracer

### This is a project to experiment with real time graphics, parallel programming, game streaming services, and much more.

This has been based off of the teachings of Peter Shirley's **"Ray Tracing In One Weekend"** series, which can be found [here](https://raytracing.github.io/).

## Build Details & Requirements

You will need a CUDA capable NVIDIA GPU with appropriate drivers and the CUDA toolkit installed. Note that this project has only been tested on a Windows PC, and Linux setups will likely vary.

Additionally, you will need the Visual Studio Build Tools for C/C++ development.

The CMakeLists options are as follows:

- `DBUILD_SERVER`: Builds the server executable
- `DBUILD_CLIENT`: Builds the client executable
- `DBUILD_LOCAL`: Builds the local executable (recommended)

By default, all options will be built unless specified.

## Video / GIF Demo To Be Posted Soon

## Upcoming features:

- Scene select
- Scene file format
- Enhanced quality for server & client setups (better compression)
