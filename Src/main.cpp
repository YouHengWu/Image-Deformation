#include <iostream>
#include <string>
#include <vector>
#include <ctime>

#include <opencv2/opencv.hpp>

#include "MLS.h"

using namespace std;
using namespace cv;

int index;
int PROGRESS;

string imgs = "image";

string Int_To_String(int x)
{
    string tmp = "";
    while (x > 0) {
        tmp += (char)(x % 10 + '0');
        x /= 10;
    }
    reverse(tmp.begin(), tmp.end());

    return tmp;
}

int main() {

    Mat image;

    int h, w, channel;

    cout << "Please select your image, input an integer between 1 to 6¡G";
    cin >> index;

    imgs += Int_To_String(index) + ".jpg";

    image = imread(imgs);
    

    cout << "Please select the deformation method(1 : affine deformation 2 : similarity deformation 3 : rigid deformation):";
    cin >> PROGRESS;

    h = image.rows, w = image.cols, channel = image.channels();

    imshow("Original", image);

    image.convertTo(image, CV_32FC3);

    float* img_data = (float*)image.data;

    vector<float> p;
    vector<float> q;

    if (index == 1) {
        //image.jpg
        p = { 40, 160, 170, 160, 310, 160, 135, 290, 230, 290, 115, 380, 255, 380 };
        q = { 40, 270, 170, 160, 330, 30, 105, 250, 200, 310, 115, 380, 255, 380 };
    }
    else if (index == 2) {
        //image2.jpg
        p = { 20, 80, 85, 80, 155, 80, 67, 145, 115, 145, 57, 190, 127, 190 };
        q = { 20, 135, 85, 80, 165, 15, 52, 125, 100, 155, 57, 190, 127, 190 };
    }
    else if (index == 3) {
        //image3.jpg
        p = {32, 127, 124, 127, 124, 60, 216, 127, 96, 204, 154, 204, 84, 260, 173, 260};
        q = {32, 87, 124, 127, 124, 60, 216, 87, 116, 204, 134, 204, 128, 260, 129, 260};
    }
    else if (index == 4) {
        //image4.jpg
        p = { 18, 125, 20, 160, 30, 244, 93, 257, 121, 254, 110, 177, 121, 129, 162, 114, 188, 102, 194, 124, 196, 150 };
        q = { 18, 125, 20, 160, 30, 244, 93, 257, 121, 254, 110, 177, 121, 129, 137, 209, 163, 197, 169, 219, 170, 245 };
    }
    else if (index == 5) {
        //image5.jpg
        p = { 39, 54, 125, 51, 57, 86, 101, 85, 38, 98, 130, 101, 64, 120, 84, 120, 60, 133, 96, 126, 47, 141, 122, 137, 78, 161, 75, 136 };
        q = { 39, 54, 125, 51, 57, 86, 101, 85, 38, 98, 130, 101, 64, 120, 84, 120, 68, 139, 87, 133, 47, 141, 122, 137, 78, 161, 75, 136 };
    }
    else if (index == 6) {
        //image6.jpg
        p = { 39, 54, 125, 51, 57, 86, 101, 85, 38, 98, 130, 101, 64, 120, 84, 120, 60, 133, 96, 126, 47, 141, 122, 137, 78, 161, 75, 136 };
        q = { 39, 54, 125, 51, 57, 86, 101, 85, 38, 98, 130, 101, 64, 120, 84, 120, 68, 139, 87, 133, 47, 141, 122, 137, 78, 161, 75, 136 };
    }
    else {
        p = {};
        q = {};
    }
    float* d_p = p.data();
    float* d_q = q.data();

    if (PROGRESS == 1)
    {
        Mat result1(h, w, CV_32FC3);

        float* Deformation_Result = (float*)result1.data;

        clock_t time1 = clock();

        Affine_Deformations_Ver2(img_data, h, w, d_p, d_q, p.size() / 2, 1.0, 1.0, Deformation_Result);

        cout << "Affine Deformations spends " << (clock() - time1) / (double)CLOCKS_PER_SEC << " seconds" << endl;

        result1.convertTo(result1, CV_8UC3);

        imshow("Affine_Transformation", result1);
        imwrite("Affine_Transformation.jpg", result1);
    }
    if (PROGRESS == 2)
    {
        Mat result2(h, w, CV_32FC3);

        float* Deformation_Result = (float*)result2.data;

        clock_t time2 = clock();

        Similarity_Deformations_Ver2(img_data, h, w, d_p, d_q, p.size() / 2, 1.0, 1.0, Deformation_Result);

        cout << "Similarity Deformations spends " << (clock() - time2) / (double)CLOCKS_PER_SEC << " seconds" << endl;

        result2.convertTo(result2, CV_8UC3);

        imshow("Similarity_Deformations", result2);
        imwrite("Similarity_Deformations.jpg", result2);
    }
    if (PROGRESS == 3)
    {
        Mat result3(h, w, CV_32FC3);

        float* Deformation_Result = (float*)result3.data;

        clock_t time3 = clock();

        Rigid_Deformations_Ver2(img_data, h, w, d_p, d_q, p.size() / 2, 1.0, 1.0, Deformation_Result);

        cout << "Rigid Deformations spends " << (clock() - time3) / (double)CLOCKS_PER_SEC << " seconds" << endl;

        result3.convertTo(result3, CV_8UC3);

        imshow("Rigid_Deformations", result3);
        imwrite("Rigid_Deformations.jpg", result3);
    }
    
    waitKey();
    return 0;
}
