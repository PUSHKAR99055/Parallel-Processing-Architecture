#include "opencv2/opencv.hpp"
#include "sssp_common.h"

#define cvCellSizeX 65
#define cvCellSizeY 15
#define cvFontSize 0.6

float traversePath(int x, cv::Mat image, int recursionCount, float *d_c, int index, pathElement *path)
{   
    int nextIndex;
    float cost;
    if(path[index].pathIndex == -1)
    {   
        printf("(%d, %d) -> ", (int)(index) / MATRIX_SIZE, index % MATRIX_SIZE);
        image.at<cv::Vec3b>((int)(index / MATRIX_SIZE)*0.9, (index % MATRIX_SIZE)*1) = cv::Vec3b(0, 255, 255);
        return d_c[index];
    };

    nextIndex = path[index].pathIndex;
    recursionCount++;
    printf("step %d, ", recursionCount);
    float val = traversePath(x, image, recursionCount, d_c, nextIndex, path);
    recursionCount--;
    cost = (recursionCount > 0 ? d_c[index] : 0) + val;                                             //the first calling function is the one with destination whose weight should not be added to cost.
    printf("(%d, %d) -> ", (int)(index / MATRIX_SIZE), index % MATRIX_SIZE);
    try
    {
        image.at<cv::Vec3b>((int)(index / MATRIX_SIZE)/(x), (index % MATRIX_SIZE)/(x)) = cv::Vec3b(0, 255, 255);
    }
    catch(const std::exception& e)
    {
        std::cerr << e.what() << '\n';
    }
    
    
    return cost;
}

void sssp(int x, float *d_c, pathElement *intermediateCostsAndPath, int destinationIndex)
{
    cv::Mat sspImage = cv::Mat::zeros(MATRIX_SIZE/(x) * 0.9, MATRIX_SIZE/(x)*1, CV_8UC3);

    printf("\n");
    int recursionCount = 0;
    float cost = traversePath(x, sspImage, recursionCount, d_c, destinationIndex, intermediateCostsAndPath);
    //printf("\ncalculated cost - %f", cost);
    cv::imshow("SSSP", sspImage);
    cv::waitKey(0);
}


extern void showSSSP(float *d_c, pathElement *intermediateCostsAndPath, int destinationIndex)
{   
    int x = 1;
    do
    {
        try
        {
            sssp(x, d_c, intermediateCostsAndPath, destinationIndex);
        }
        catch(const std::exception& e)
        {
            std::cerr << e.what() << '\n';
        }  
        std::cout << "\nEnter OpenCV image scale factor - ";
        std::cin >> x;
        
    }while(x != -1);
    return;
}