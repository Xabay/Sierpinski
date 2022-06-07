#include <sstream>
#include <fstream>
#include <algorithm>
#include <memory>
#include <iostream>
#include <vector>
#include <CL\cl.h>
#include <chrono>
#include <thread>

#define NOMINMAX
template<typename T> T min(T const& x, T const& y){ return x<y?x:y; }
template<typename T> T max(T const& x, T const& y){ return x<y?y:x; }
#include <windows.h>
#include <Windowsx.h>
#include <WinUser.h>
#include <gdiplus.h>

using namespace Gdiplus;
#pragma comment (lib,"Gdiplus.lib")

static int GDIPlus_GetEncoderClsid(const wchar_t* format, CLSID* pClsid)
{
	unsigned int num = 0;          // number of image encoders
	unsigned int size = 0;         // size of the image encoder array in bytes
	ImageCodecInfo* pImageCodecInfo = nullptr;
		
	GetImageEncodersSize(&num, &size);
	if(size == 0){ return -1; }  // Failure
		
	pImageCodecInfo = (ImageCodecInfo*)(new unsigned char[size]);
	if(!pImageCodecInfo){ return -1; } // Failure
		
	GetImageEncoders(num, size, pImageCodecInfo);
	for(int j=0; j<(int)num; ++j)
	{
		if( wcscmp(pImageCodecInfo[j].MimeType, format) == 0 )
		{
			*pClsid = pImageCodecInfo[j].Clsid;
			delete[] pImageCodecInfo;
			return j;  // Success
		}
	}
	delete[] pImageCodecInfo;
	return -1; // Failure
}

std::vector<float> sierpinski_simple(std::vector<float> const& prevIter, size_t W, size_t H, int depth) {

	//if at the end of recursion return with the result
	if(depth <= 0) {
		return prevIter;
	}
	std::vector<float> nextIter(W*H*4);
	auto start_t = std::chrono::high_resolution_clock::now();

	//iterate through the image data
	for(int y=0; y<H; y+=2)
	{
		for(int x=0; x<W; x+=2)
		{
			//get a,r,g,b values for first pixel in every 2x2 block
			float a = prevIter[(y*W + x) * 4 + 0];
			float r = prevIter[(y*W + x) * 4 + 1];
			float g = prevIter[(y*W + x) * 4 + 2];
			float b = prevIter[(y*W + x) * 4 + 3];

			//set the read values to shrunk duplicate in bottom left
			nextIter[((y/2 + H/2)*W +  x/2) * 4 + 0] = a;
			nextIter[((y/2 + H/2)*W +  x/2) * 4 + 1] = r;
			nextIter[((y/2 + H/2)*W +  x/2) * 4 + 2] = g;
			nextIter[((y/2 + H/2)*W +  x/2) * 4 + 3] = b;

			//bottom right
			nextIter[((y/2 + H/2)*W +  x/2 + W/2) * 4 + 0] = a;
			nextIter[((y/2 + H/2)*W +  x/2 + W/2) * 4 + 1] = r;
			nextIter[((y/2 + H/2)*W +  x/2 + W/2) * 4 + 2] = g;
			nextIter[((y/2 + H/2)*W +  x/2 + W/2) * 4 + 3] = b;

			//top middle
			nextIter[((y/2)*W +  x/2 + W/4) * 4 + 0] = a;
			nextIter[((y/2)*W +  x/2 + W/4) * 4 + 1] = r;
			nextIter[((y/2)*W +  x/2 + W/4) * 4 + 2] = g;
			nextIter[((y/2)*W +  x/2 + W/4) * 4 + 3] = b;
		}
	}

	auto end_t = std::chrono::high_resolution_clock::now();
	auto t_cpu = std::chrono::duration_cast<std::chrono::microseconds>(end_t - start_t).count()/1000.0;
	std::cout<<"Current iteration took " << t_cpu << " msecs on naive cpu implementation.\n";
	return sierpinski_simple(nextIter, W, H, depth-1);
}

std::vector<float> sierpinski_concurrent(std::vector<float> const& prevIter, size_t W, size_t H, int depth) {
	if(depth <= 0) {
		return prevIter;
	}
	
	std::vector<float> nextIter(W*H*4);
	std::vector<std::thread> threads;
	auto start_t = std::chrono::high_resolution_clock::now();
	auto cores = std::thread::hardware_concurrency();
	for (int i=0; i<cores; ++i) {

		//divide picture into vertical stripes based on the number of available threads
		int y_start = i*H/cores;
		int y_end = (i+1)*H/cores;

		
		//process each stripe with separate thread
		std::thread new_thread ([](std::vector<float> const& prevIter, std::vector<float>& nextIter, size_t W, size_t H, int start, int end) {
			for(int y=start; y<end; y+=2)
				{
					for(int x=0; x<W; x+=2)
					{
						float a = prevIter[(y*W + x) * 4 + 0];
						float r = prevIter[(y*W + x) * 4 + 1];
						float g = prevIter[(y*W + x) * 4 + 2];
						float b = prevIter[(y*W + x) * 4 + 3];

						nextIter[((y/2 + H/2)*W +  x/2) * 4 + 0] = a;
						nextIter[((y/2 + H/2)*W +  x/2) * 4 + 1] = r;
						nextIter[((y/2 + H/2)*W +  x/2) * 4 + 2] = g;
						nextIter[((y/2 + H/2)*W +  x/2) * 4 + 3] = b;

						nextIter[((y/2 + H/2)*W +  x/2 + W/2) * 4 + 0] = a;
						nextIter[((y/2 + H/2)*W +  x/2 + W/2) * 4 + 1] = r;
						nextIter[((y/2 + H/2)*W +  x/2 + W/2) * 4 + 2] = g;
						nextIter[((y/2 + H/2)*W +  x/2 + W/2) * 4 + 3] = b;

						nextIter[((y/2)*W +  x/2 + W/4) * 4 + 0] = a;
						nextIter[((y/2)*W +  x/2 + W/4) * 4 + 1] = r;
						nextIter[((y/2)*W +  x/2 + W/4) * 4 + 2] = g;
						nextIter[((y/2)*W +  x/2 + W/4) * 4 + 3] = b;
					}
				}
		},std::cref(prevIter), std::ref(nextIter), W, H, y_start, y_end);

		threads.push_back(std::move(new_thread));
	}
	
	//wait for all the threads
	for(int i=0; i<cores; ++i) {
		threads[i].join();
	}
	

	auto end_t = std::chrono::high_resolution_clock::now();
	auto t_cpu = std::chrono::duration_cast<std::chrono::microseconds>(end_t - start_t).count()/1000.0;
	std::cout<<"Current iteration took " << t_cpu << " msecs on concurrent cpu implementation.\n";
	return sierpinski_concurrent(nextIter, W, H, depth-1);
}

int main()
{
    GdiplusStartupInput gdiplusStartupInput;
	ULONG_PTR gdiplusToken;
    GdiplusStartup(&gdiplusToken, &gdiplusStartupInput, nullptr);
    

    Bitmap image( L"input.png" );
    size_t W = image.GetWidth();
    size_t H = image.GetHeight();

	cl_platform_id platform = NULL;
	auto status = clGetPlatformIDs(1, &platform, NULL);
	
	cl_device_id device = NULL;
	status = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device, NULL);
	if (status != CL_SUCCESS){ printf("Failed to get device!\n"); }

	cl_context_properties cps[] = { CL_CONTEXT_PLATFORM, (cl_context_properties)platform, 0 };
	auto context = clCreateContext(cps, 1, &device, 0, 0, &status);
	if (status != CL_SUCCESS){ printf("Context creation failed!\n"); }

	auto queue = clCreateCommandQueue(context, device, CL_QUEUE_PROFILING_ENABLE, &status);
	if (status != CL_SUCCESS){ printf("Queue creation failed!\n"); }

	std::ifstream file("sierpinski.cl");
	std::string source( std::istreambuf_iterator<char>(file), (std::istreambuf_iterator<char>()));
	size_t      sourceSize = source.size();
	const char* sourcePtr  = source.c_str();
	auto program = clCreateProgramWithSource(context, 1, &sourcePtr, &sourceSize, &status);
	if (status != CL_SUCCESS){ printf("Program creation failed!\n"); }
	
	status = clBuildProgram(program, 1, &device, "", nullptr, nullptr);
	if (status != CL_SUCCESS)
	{
		size_t len = 0;
		status = clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, 0, nullptr, &len);
		std::unique_ptr<char[]> log = std::make_unique<char[]>(len);
		status = clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, len, log.get(), nullptr);
		std::cout << log.get() << "\n";
		return -1;
	}

	auto kernel = clCreateKernel(program, "sierpinski", &status);
	if (status != CL_SUCCESS){ printf("Kernel creation failed!\n"); }

	
    std::vector<float> image_data(W*H*4);
	std::vector<float> image_data_cpu(W*H*4);
	std::vector<float> image_data_concurrent(W*H*4);
	std::vector<float> empty_data(W*H*4, 0);
    
    Rect rct; rct.X = 0; rct.Y = 0; rct.Width = W; rct.Height = H;
    BitmapData bmpdata;
	
    if( image.LockBits(&rct, ImageLockModeRead, PixelFormat32bppARGB, &bmpdata) == Status::Ok)
    {
        for(int y=0; y<H; ++y)
        {
            for(int x=0; x<W; ++x)
            {
                auto p = ((Color*)bmpdata.Scan0)[y * bmpdata.Stride / 4 + x];
                image_data[(y*W+x)*4 + 0] = (float)p.GetRed() / 255.0f;
                image_data[(y*W+x)*4 + 1] = (float)p.GetGreen() / 255.0f;
                image_data[(y*W+x)*4 + 2] = (float)p.GetBlue() / 255.0f;
                image_data[(y*W+x)*4 + 3] = (float)p.GetAlpha() / 255.0f;
            }
        }
    }
    image.UnlockBits(&bmpdata);
    
	
	
    cl_image_format format = { CL_RGBA, CL_FLOAT };
	
	cl_mem img_src, img_dst;
	int lvl = 9;
	
	image_data_cpu = sierpinski_simple(std::cref(image_data), W, H, lvl);
	image_data_concurrent = sierpinski_concurrent(std::cref(image_data), W, H, lvl);



	img_src = clCreateImage2D(context, CL_MEM_READ_WRITE  | CL_MEM_USE_HOST_PTR, &format, W, H, 0, image_data.data(), &status);
	if (status != CL_SUCCESS){ printf("Image allocation failed!\n"); }
	img_dst = clCreateImage2D(context, CL_MEM_READ_WRITE | CL_MEM_USE_HOST_PTR, &format, W, H, 0, empty_data.data(), &status);
	if (status != CL_SUCCESS){ printf("Image allocation failed!\n"); }

	

	for(int i=0; i<lvl; ++i) {

		status = clSetKernelArg(kernel, 0, sizeof(img_src), &img_src);
		if (status != CL_SUCCESS){ printf("Failed to set first kernel arg!\n"); }
		status = clSetKernelArg(kernel, 1, sizeof(img_dst), &img_dst);
		if (status != CL_SUCCESS){ printf("Failed to set second kernel arg!\n"); }

		cl_event ev;
		cl_ulong t_start, t_end;


		size_t kernel_dims[2] = {W, H};
		status = clEnqueueNDRangeKernel(queue, kernel, 2, nullptr, kernel_dims, nullptr, 0, nullptr, &ev);
		if (status != CL_SUCCESS){ printf("Failed to enqueue command!\n"); }
		status = clWaitForEvents(1, &ev);
		if (status != CL_SUCCESS){ printf("Failed to wait for event!\n"); }
		status = clGetEventProfilingInfo(ev, CL_PROFILING_COMMAND_START, sizeof(t_start), &t_start, nullptr);
		if (status != CL_SUCCESS){ printf("Failed to get start event info!\n"); }
		status = clGetEventProfilingInfo(ev, CL_PROFILING_COMMAND_END, sizeof(t_end), &t_end, nullptr);
		if (status != CL_SUCCESS){ printf("Failed to get end event info!\n"); }

		clReleaseEvent(ev);

		size_t origin[3] = {0, 0, 0};
		size_t dims[3] = {W, H, 1};
		status = clEnqueueReadImage(queue, img_dst, false, origin, dims, 0, 0, image_data.data(), 0, nullptr, nullptr);
		if (status != CL_SUCCESS){ printf("Reading image failed!\n"); }

		status = clEnqueueCopyImage(queue, img_dst, img_src, origin, origin, dims, 0, nullptr, nullptr);
		if (status != CL_SUCCESS){ printf("Failed to copy image!\n"); }

		status = clFinish(queue);
		if (status != CL_SUCCESS){ printf("Failed to wait for the queue!\n"); }
		

		

		std::cout<<"Current iteration took " << (t_end - t_start) * 0.001 * 0.001 << " msecs on gpu implementation.\n";
	}
	
	

	
    CLSID Clsid;
	GDIPlus_GetEncoderClsid(L"image/png", &Clsid);
	
    Bitmap& bmp = Bitmap(W, H, PixelFormat32bppARGB);

    BitmapData bmpdata2;
    if( bmp.LockBits(&rct, ImageLockModeWrite, PixelFormat32bppARGB, &bmpdata2) == Status::Ok )
    {
        for(int y=0; y<H; ++y)
        {
            for(int x=0; x<W; ++x)
            {
                auto r = (BYTE)(image_data[(y*W+x)*4 + 0] * 255.0);
                auto g = (BYTE)(image_data[(y*W+x)*4 + 1] * 255.0);
                auto b = (BYTE)(image_data[(y*W+x)*4 + 2] * 255.0);
                auto a = (BYTE)(image_data[(y*W+x)*4 + 3] * 255.0);

                ((Color*)bmpdata2.Scan0)[y * bmpdata2.Stride / 4 + x] = Color(a, r, g, b);
            }
        }
    }
    bmp.UnlockBits(&bmpdata2);
    bmp.Save( L"output_gpu.png", &Clsid );

    if( bmp.LockBits(&rct, ImageLockModeWrite, PixelFormat32bppARGB, &bmpdata2) == Status::Ok )
    {
        for(int y=0; y<H; ++y)
        {
            for(int x=0; x<W; ++x)
            {
                auto r = (BYTE)(image_data_cpu[(y*W+x)*4 + 0] * 255.0);
                auto g = (BYTE)(image_data_cpu[(y*W+x)*4 + 1] * 255.0);
                auto b = (BYTE)(image_data_cpu[(y*W+x)*4 + 2] * 255.0);
                auto a = (BYTE)(image_data_cpu[(y*W+x)*4 + 3] * 255.0);

                ((Color*)bmpdata2.Scan0)[y * bmpdata2.Stride / 4 + x] = Color(a, r, g, b);
            }
        }
    }
    bmp.UnlockBits(&bmpdata2);
    bmp.Save( L"output_cpu.png", &Clsid );


    if( bmp.LockBits(&rct, ImageLockModeWrite, PixelFormat32bppARGB, &bmpdata2) == Status::Ok )
    {
        for(int y=0; y<H; ++y)
        {
            for(int x=0; x<W; ++x)
            {
                auto r = (BYTE)(image_data_concurrent[(y*W+x)*4 + 0] * 255.0);
                auto g = (BYTE)(image_data_concurrent[(y*W+x)*4 + 1] * 255.0);
                auto b = (BYTE)(image_data_concurrent[(y*W+x)*4 + 2] * 255.0);
                auto a = (BYTE)(image_data_concurrent[(y*W+x)*4 + 3] * 255.0);

                ((Color*)bmpdata2.Scan0)[y * bmpdata2.Stride / 4 + x] = Color(a, r, g, b);
            }
        }
    }
    bmp.UnlockBits(&bmpdata2);
    bmp.Save( L"output_concurrent.png", &Clsid );


    clReleaseMemObject(img_src);
    clReleaseMemObject(img_dst);
    clReleaseKernel(kernel);
    clReleaseProgram(program);
    clReleaseCommandQueue(queue);
    clReleaseContext(context);
    //clReleaseDevice(device);  //ez a fuggveny nincs az en openCL verziomban (azt talaltam az interneten, hogy az nVidia eszkozokhoz csak 1.2 van 2.0 helyett)
	
	
	return 0;
}

