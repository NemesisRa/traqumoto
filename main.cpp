#include <stdio.h>
#include <stdlib.h>
#include <opencv2/opencv.hpp>


using namespace std;
using namespace cv;

int main (int argc, char* argv[])
{
    CvCapture *vid = 0;
    vid = cvCaptureFromAVI("/home/eleves/Vidéos/video.avi");
    if(!vid){
        cerr << "Video pas ouverte" << endl;
        return 1;
    }

    cout << "'q' pour quitter..." << endl;
    int key = 0;
    int H_MIN,S_MIN,V_MIN,H_MAX,S_MAX,V_MAX;
    IplImage *img,*imgHSV;
    cvNamedWindow ("Image", CV_WINDOW_AUTOSIZE);

    while(true)
    {
        img = cvQueryFrame(vid);
        if(!img) break;
        if(key == 1048689 || key==1048603) break;

        cvShowImage ("Image", img);
        key = cvWaitKey(42);
    }
    cvDestroyAllWindows();
    cvReleaseImage(&img);
    return 0;
}