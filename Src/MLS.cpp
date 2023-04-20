#include "MLS.h"
#include <iostream>
#include <algorithm>
#include <string>
#include <cmath>

#include <Eigen/Core>
#include <Eigen/Dense>

using namespace std;
using namespace Eigen;

void linspace(float start, float end, int num, float *res)
{
    float delta = (end - start) / num;

    for (int i = 0; i < num; ++i) 
    {
        res[i] = start + i * delta;
    }
}

// h = 352, w = 409, p.size() = 14, q.size() = 14, Control_Points = 7, alpha = 1.0, density = 1.0
void Affine_Deformations(float *image, int h, int w, float *p, float *q, int Control_Points, float alpha, float density, float *result)
{
    vector<float> gridX(w);
    vector<float> gridY(h);

    float *d_gridX = gridX.data();
    float *d_gridY = gridY.data();

    linspace(0, w, w, d_gridX);
    linspace(0, h, h, d_gridY);

    int row = gridY.size(), column = gridX.size();

    float *Point_X = (float*) malloc(row * column * sizeof(float));
    float *Point_Y = (float*) malloc(row * column * sizeof(float));

    // Point_X Point_Y：相片座標位置
    for(int i = 0; i < row * column; ++i)
    {
        int u = i % column;
        int v = i / column;

        Point_X[i] = d_gridX[u];//column
        Point_Y[i] = d_gridY[v];//row
    }

    //d_w => w(i)
    float *d_w = (float *) malloc(Control_Points * row * column * sizeof(float));

    for (int i = 0; i < row * column; ++i)
    {
        for (int j = 0; j < Control_Points; ++j)
        {
            d_w[row * column * j + i] = 1.0 / pow(sqrt((pow(p[2 * j] - Point_X[i], 2) + pow(p[2 * j + 1] - Point_Y[i], 2))), 2 * alpha);
        }
    }

    //Pstar, Qstar 前row * column 是 x 後 row * column 是 y
    float *Pstar = (float *) malloc(row * column * 2 * sizeof(float));
    float *Qstar = (float *) malloc(row * column * 2 * sizeof(float));

    //Pstar = p*, Qstar = q*
    for(int i = 0; i < row * column; ++i) 
    {
        float w_px = 0, w_py = 0, w_tmp = 0;
        float w_qx = 0, w_qy = 0;

        for(int j = 0; j < Control_Points; ++j) 
        {
            w_tmp += d_w[j * row * column + i];
	        w_px += d_w[j * row * column + i] * p[2 * j];
	        w_py += d_w[j * row * column + i] * p[2 * j + 1];
	        w_qx += d_w[j * row * column + i] * q[2 * j];
	        w_qy += d_w[j * row * column + i] * q[2 * j + 1];
        }
        Pstar[i] = w_px / w_tmp;
        Pstar[i + row * column] = w_py / w_tmp;
        Qstar[i] = w_qx / w_tmp;
        Qstar[i + row * column] = w_qy / w_tmp;
    }

    float *Phat = (float *) malloc(row * column * 2 * Control_Points * sizeof(float));
    float *Qhat = (float *) malloc(row * column * 2 * Control_Points * sizeof(float));

    //Phat = p^, Qhat = q^
    //p^ = pi - p*, q^ = qi - q*

    for(int i = 0; i < row * column; ++i) 
    {
        for(int j = 0; j < Control_Points; ++j) 
        {
            Phat[j * row * column * 2 + i] = p[2 * j] - Pstar[i];
	        Phat[j * row * column * 2 + row * column + i] = p[2 * j + 1] - Pstar[i + row * column];

            Qhat[j * row * column * 2 + i] = q[2 * j] - Qstar[i];
	        Qhat[j * row * column * 2 + row * column + i] = q[2 * j + 1] - Qstar[i + row * column];
        }
    }

    //pTwp：all the 2 * 2 matrix
    //pT：transpose of Phat
    float *pTwp = (float*) malloc(row * column * 4 * sizeof(float));

    for(int i = 0; i < row * column; ++i) 
    {
        float tmp1 = 0, tmp2 = 0, tmp3 = 0, tmp4 = 0;

        for(int j = 0; j < Control_Points; ++j) 
        {
            tmp1 += (Phat[j * row * column * 2 + i] * Phat[j * row * column * 2 + i] * d_w[j * row * column + i]);
            tmp2 += (Phat[j * row * column * 2 + i] * Phat[j * row * column * 2 + row * column + i] * d_w[j * row * column + i]);
            tmp3 += (Phat[j * row * column * 2 + row * column + i] * Phat[j * row * column * 2 + i] * d_w[j * row * column + i]);
            tmp4 += (Phat[j * row * column * 2 + row * column + i] * Phat[j * row * column * 2 + row * column + i] * d_w[j * row * column + i]);
        }
        pTwp[4 * i] = tmp1;
        pTwp[4 * i + 1] = tmp2;
        pTwp[4 * i + 2] = tmp3;
        pTwp[4 * i + 3] = tmp4;
        
    }

    //pTwp：all the 2 * 2 matrix
    //pT：transpose of Phat
    float *pTwq = (float*) malloc(row * column * 4 * sizeof(float));

    for(int i = 0; i < row * column; ++i) 
    {
        float tmp1 = 0, tmp2 = 0, tmp3 = 0, tmp4 = 0;

        for(int j = 0; j < Control_Points; ++j) 
        {
            tmp1 += (Phat[j * row * column * 2 + i] * Qhat[j * row * column * 2 + i] * d_w[j * row * column + i]);
            tmp2 += (Phat[j * row * column * 2 + i] * Qhat[j * row * column * 2 + row * column + i] * d_w[j * row * column + i]);
            tmp3 += (Phat[j * row * column * 2 + row * column + i] * Qhat[j * row * column * 2 + i] * d_w[j * row * column + i]);
            tmp4 += (Phat[j * row * column * 2 + row * column + i] * Qhat[j * row * column * 2 + row * column + i] * d_w[j * row * column + i]);
        }
        pTwq[4 * i] = tmp1;
        pTwq[4 * i + 1] = tmp2;
        pTwq[4 * i + 2] = tmp3;
        pTwq[4 * i + 3] = tmp4;
    }
   
    float *d_m = (float*) malloc(row * column * 4 * sizeof(float));

    for(int i = 0; i < row * column; ++i) 
    {        
        /*
        Matrix2f pwq, pwq_inv, pwp, pwp_inv, tmp_m;

        pwq(0, 0) = pTwq[4 * i]; 
        pwq(0, 1) = pTwq[4 * i + 1]; 
        pwq(1, 0) = pTwq[4 * i + 2]; 
        pwq(1, 1) = pTwq[4 * i + 3];
        //pwq_inv = pwq.inverse();

        pwp(0, 0) = pTwp[4 * i];
        pwp(0, 1) = pTwp[4 * i + 1]; 
        pwp(1, 0) = pTwp[4 * i + 2];
        pwp(1, 1) = pTwp[4 * i + 3];
        pwp_inv = pwp.inverse();
        */
        //tmp_m = pwp_inv * pwq;
        //tmp_m = pwq_inv * pwp;
        /*
        d_m[4 * i] = tmp_m(0,0);
        d_m[4 * i + 1] = tmp_m(0,1);
        d_m[4 * i + 2] = tmp_m(1,0);
        d_m[4 * i + 3] = tmp_m(1,1);
        */
        float ad_bc = pTwp[4 * i] * pTwp[4 * i + 3] - pTwp[4 * i + 1] * pTwp[4 * i + 2];

        d_m[4 * i] = (float)(1.0f / ad_bc) * (pTwp[4 * i + 3] * pTwq[4 * i] - pTwp[4 * i + 1] * pTwq[4 * i + 2]);
        d_m[4 * i + 1] = (float)(1.0f / ad_bc) * (pTwp[4 * i + 3] * pTwq[4 * i + 1] - pTwp[4 * i + 1] * pTwq[4 * i + 3]);
        d_m[4 * i + 2] = (float)(1.0f / ad_bc) * (-pTwp[4 * i + 2] * pTwq[4 * i] + pTwp[4 * i] * pTwq[4 * i + 2]);
        d_m[4 * i + 3] = (float)(1.0f / ad_bc) * (-pTwp[4 * i + 2] * pTwq[4 * i + 1] + pTwp[4 * i] * pTwq[4 * i + 3]);
    }

    float *transform = (float*) malloc(row * column * 2 * sizeof(float));

    for(int i = 0; i < row * column; ++i) 
    {
        transform[2 * i] = (Point_X[i] - Pstar[i]) * d_m[4 * i] + (Point_Y[i] - Pstar[i + row * column]) * d_m[4 * i + 2] + Qstar[i];
        transform[2 * i + 1] = (Point_X[i] - Pstar[i]) * d_m[4 * i + 1] + (Point_Y[i] - Pstar[i + row * column]) * d_m[4 * i + 3] + Qstar[i + row * column];

        //transform[2 * i] = (Point_X[i] - Qstar[i]) * d_m[4 * i] + (Point_Y[i] - Qstar[i + row * column]) * d_m[4 * i + 2] + Pstar[i];
        //transform[2 * i + 1] = (Point_X[i] - Qstar[i]) * d_m[4 * i + 1] + (Point_Y[i] - Qstar[i + row * column]) * d_m[4 * i + 3] + Pstar[i + row * column];
        
    }
    //Bounding box check
    for(int i = 0; i < row * column * 2; ++i) 
    {
        if (!(transform[i] >= 0) || (transform[i] > h - 1 && i % 2 == 1) || (transform[i] > w - 1 && i % 2 == 0)) 
        {
            transform[i] = 0;
        }
    }
   
    for(int i = 0; i < row * column; ++i) 
    {
        int u = i % column;
        int v = i / column;

        //result(r, g, b) 三通道所以要三個 (transform x, y)
        result[3 * (v * w + u)] = image[3 * (int(transform[2 * i + 1]) * w + int(transform[2 * i]))]; 
        result[3 * (v * w + u) + 1] = image[3 * (int(transform[2 * i + 1]) * w + int(transform[2 * i])) + 1]; 
        result[3 * (v * w + u) + 2] = image[3 * (int(transform[2 * i + 1]) * w + int(transform[2 * i])) + 2]; 
    }

}

void Affine_Deformations_Ver2(float* image, int h, int w, float* p, float* q, int Control_Points, float alpha, float density, float* result)
{
    vector<float> gridX(w);
    vector<float> gridY(h);

    float* d_gridX = gridX.data();
    float* d_gridY = gridY.data();

    linspace(0, w, w, d_gridX);
    linspace(0, h, h, d_gridY);

    int row = gridY.size(), column = gridX.size();

    float **Point_X = new float*[row];
    float **Point_Y = new float*[row];

    for (int i = 0; i < row; ++i)
    {
        Point_X[i] = new float[column];
        Point_Y[i] = new float[column];
    }
    
    //float* Point_X = (float*)malloc(row * column * sizeof(float));
    //float* Point_Y = (float*)malloc(row * column * sizeof(float));
    

    // Point_X Point_Y：相片座標位置
    for (int i = 0; i < row; ++i)
    {
        for (int j = 0; j < column; ++j)
        {
            int u = (i * column + j) % column;
            int v = (i * column + j) / column;

            Point_X[i][j] = d_gridX[u];//column
            Point_Y[i][j] = d_gridY[v];//row
        }
    }

    //d_w => w(i)
    //float* d_w = (float*)malloc(Control_Points * row * column * sizeof(float));
    float*** d_w = new float** [Control_Points];

    for (int i = 0; i < Control_Points; ++i)
    {
        d_w[i] = new float* [row];
        for(int j = 0; j < row; ++j)
        {
            d_w[i][j] = new float[column];
        }
    }

    for (int i = 0; i < Control_Points; ++i)
    {
        for (int j = 0; j < row; ++j)
        {
            for (int k = 0; k < column; ++k) {
                d_w[i][j][k] = 1.0 / pow(sqrt((pow(p[2 * i] - Point_X[j][k], 2) + pow(p[2 * i + 1] - Point_Y[j][k], 2))), 2 * alpha);
            }
        }
    }

    //Pstar, Qstar 前row * column 是 x 後 row * column 是 y
    //float* Pstar = (float*)malloc(row * column * 2 * sizeof(float));
    //float* Qstar = (float*)malloc(row * column * 2 * sizeof(float));

    float*** Pstar = new float** [2];
    float*** Qstar = new float** [2];

    for (int i = 0; i < 2; ++i)
    {
        Pstar[i] = new float* [row];
        Qstar[i] = new float* [row];
        for (int j = 0; j < row; ++j)
        {
            Pstar[i][j] = new float[column];
            Qstar[i][j] = new float[column];
        }
    }

    //Pstar = p*, Qstar = q*
    for (int i = 0; i < row; ++i)
    {
        for (int j = 0; j < column; ++j) {

            float w_px = 0, w_py = 0, w_tmp = 0;
            float w_qx = 0, w_qy = 0;

            for (int k = 0; k < Control_Points; ++k)
            {
                w_tmp += d_w[k][i][j];
                w_px += d_w[k][i][j] * p[2 * k];
                w_py += d_w[k][i][j] * p[2 * k + 1];
                w_qx += d_w[k][i][j] * q[2 * k];
                w_qy += d_w[k][i][j] * q[2 * k + 1];
            }
            Pstar[0][i][j] = w_px / w_tmp;
            Pstar[1][i][j] = w_py / w_tmp;
            Qstar[0][i][j] = w_qx / w_tmp;
            Qstar[1][i][j] = w_qy / w_tmp;
        }
    }

    //float* Phat = (float*)malloc(row * column * 2 * Control_Points * sizeof(float));
    //float* Qhat = (float*)malloc(row * column * 2 * Control_Points * sizeof(float));

    //Phat = p^, Qhat = q^
    //p^ = pi - p*, q^ = qi - q*

    float**** Phat = new float*** [2];
    float**** Qhat = new float*** [2];

    for (int i = 0; i < 2; ++i)
    {
        Phat[i] = new float** [Control_Points];
        Qhat[i] = new float** [Control_Points];
        for (int j = 0; j < Control_Points; ++j)
        {
            Phat[i][j] = new float* [row];
            Qhat[i][j] = new float* [row];
            for (int k = 0; k < row; ++k)
            {
                Phat[i][j][k] = new float [column];
                Qhat[i][j][k] = new float [column];
            }
        }
    }

    for (int i = 0; i < row; ++i)
    {
        for (int j = 0; j < column; ++j)
        {
            for (int k = 0; k < Control_Points; ++k)
            {
                Phat[0][k][i][j] = p[2 * k] - Pstar[0][i][j];
                Phat[1][k][i][j] = p[2 * k + 1] - Pstar[1][i][j];

                Qhat[0][k][i][j] = q[2 * k] - Qstar[0][i][j];
                Qhat[1][k][i][j] = q[2 * k + 1] - Qstar[1][i][j];
            }
        }
    }

    //pTwp：all the 2 * 2 matrix
    //pT：transpose of Phat
    float**** pTwp = new float*** [2];

    for (int i = 0; i < 2; ++i)
    {
        pTwp[i] = new float** [2];
        for (int j = 0; j < 2; ++j)
        {
            pTwp[i][j] = new float* [row];
            for (int k = 0; k < row; ++k)
            {
                pTwp[i][j][k] = new float[column];
            }
        }
    }

    for (int i = 0; i < row; ++i)
    {
        for (int j = 0; j < column; ++j) {
         
            float tmp1 = 0, tmp2 = 0, tmp3 = 0, tmp4 = 0;

            for (int k = 0; k < Control_Points; ++k)
            {
                tmp1 += Phat[0][k][i][j] * Phat[0][k][i][j] * d_w[k][i][j];
                tmp2 += Phat[0][k][i][j] * Phat[1][k][i][j] * d_w[k][i][j];
                tmp3 += Phat[1][k][i][j] * Phat[0][k][i][j] * d_w[k][i][j];
                tmp4 += Phat[1][k][i][j] * Phat[1][k][i][j] * d_w[k][i][j];
            }
            pTwp[0][0][i][j] = tmp1;
            pTwp[0][1][i][j] = tmp2;
            pTwp[1][0][i][j] = tmp3;
            pTwp[1][1][i][j] = tmp4;
        }
    }

    //pTwp：all the 2 * 2 matrix
    //pT：transpose of Phat
    //float* pTwq = (float*)malloc(row * column * 4 * sizeof(float));

    float**** pTwq = new float*** [2];

    for (int i = 0; i < 2; ++i)
    {
        pTwq[i] = new float** [2];
        for (int j = 0; j < 2; ++j)
        {
            pTwq[i][j] = new float* [row];
            for (int k = 0; k < row; ++k)
            {
                pTwq[i][j][k] = new float[column];
            }
        }
    }

    for (int i = 0; i < row; ++i)
    {
        for (int j = 0; j < column; ++j) {

            float tmp1 = 0, tmp2 = 0, tmp3 = 0, tmp4 = 0;

            for (int k = 0; k < Control_Points; ++k)
            {
                tmp1 += Phat[0][k][i][j] * Qhat[0][k][i][j] * d_w[k][i][j];
                tmp2 += Phat[0][k][i][j] * Qhat[1][k][i][j] * d_w[k][i][j];
                tmp3 += Phat[1][k][i][j] * Qhat[0][k][i][j] * d_w[k][i][j];
                tmp4 += Phat[1][k][i][j] * Qhat[1][k][i][j] * d_w[k][i][j];
            }
            pTwq[0][0][i][j] = tmp1;
            pTwq[0][1][i][j] = tmp2;
            pTwq[1][0][i][j] = tmp3;
            pTwq[1][1][i][j] = tmp4;
        }
    }

    //float* d_m = (float*)malloc(row * column * 4 * sizeof(float));

    float**** d_m = new float*** [2];

    for (int i = 0; i < 2; ++i)
    {
        d_m[i] = new float** [2];
        for (int j = 0; j < 2; ++j)
        {
            d_m[i][j] = new float* [row];
            for (int k = 0; k < row; ++k)
            {
                d_m[i][j][k] = new float[column];
            }
        }
    }

    for (int i = 0; i < row; ++i)
    {
        for (int j = 0; j < column; ++j) {
            float ad_bc = pTwp[0][0][i][j] * pTwp[1][1][i][j] - pTwp[0][1][i][j] * pTwp[1][0][i][j];

            d_m[0][0][i][j] = (float)(1.0f / ad_bc) * (pTwp[1][1][i][j] * pTwq[0][0][i][j] - pTwp[0][1][i][j] * pTwq[1][0][i][j]);
            d_m[0][1][i][j] = (float)(1.0f / ad_bc) * (pTwp[1][1][i][j] * pTwq[0][1][i][j] - pTwp[0][1][i][j] * pTwq[1][1][i][j]);
            d_m[1][0][i][j] = (float)(1.0f / ad_bc) * (-pTwp[1][0][i][j] * pTwq[0][0][i][j] + pTwp[0][0][i][j] * pTwq[1][0][i][j]);
            d_m[1][1][i][j] = (float)(1.0f / ad_bc) * (-pTwp[1][0][i][j] * pTwq[0][1][i][j] + pTwp[0][0][i][j] * pTwq[1][1][i][j]);
        }
    }

    //float* transform = (float*)malloc(row * column * 2 * sizeof(float));

    float*** transform = new float** [2];

    for (int i = 0; i < 2; ++i)
    {
        transform[i] = new float* [row];
        for (int j = 0; j < row; ++j)
        {
            transform[i][j] = new float[column];
        }
    }

    for (int i = 0; i < row; ++i)
    {
        for (int j = 0; j < column; ++j) {
            transform[0][i][j] = (Point_X[i][j] - Pstar[0][i][j]) * d_m[0][0][i][j] + (Point_Y[i][j] - Pstar[1][i][j]) * d_m[1][0][i][j] + Qstar[0][i][j];
            transform[1][i][j] = (Point_X[i][j] - Pstar[0][i][j]) * d_m[0][1][i][j] + (Point_Y[i][j] - Pstar[1][i][j]) * d_m[1][1][i][j] + Qstar[1][i][j];

            //transform[2 * i] = (Point_X[i] - Qstar[i]) * d_m[4 * i] + (Point_Y[i] - Qstar[i + row * column]) * d_m[4 * i + 2] + Pstar[i];
            //transform[2 * i + 1] = (Point_X[i] - Qstar[i]) * d_m[4 * i + 1] + (Point_Y[i] - Qstar[i + row * column]) * d_m[4 * i + 3] + Pstar[i + row * column];
        }
    }
    //Bounding box check
    for (int i = 0; i < row; ++i)
    {
        for (int j = 0; j < column; ++j)
        {
            for (int k = 0; k < 2; ++k)
            {
                if (!(transform[k][i][j] >= 0) || (transform[k][i][j] > h - 1 && (i * 2 * column + j * 2 + k) % 2 == 1) || (transform[k][i][j] > w - 1 && (i * 2 * column + j * 2 + k) % 2 == 0))
                {
                    transform[k][i][j] = 0;
                }
            }
        }
    }

    for (int i = 0; i < row * column; ++i)
    {
        int u = i % column;
        int v = i / column;

        //result(r, g, b) 三通道所以要三個 (transform x, y)
        result[3 * (v * w + u)] = image[3 * (int(transform[1][v][u]) * w + int(transform[0][v][u]))];
        result[3 * (v * w + u) + 1] = image[3 * (int(transform[1][v][u]) * w + int(transform[0][v][u])) + 1];
        result[3 * (v * w + u) + 2] = image[3 * (int(transform[1][v][u]) * w + int(transform[0][v][u])) + 2];
    }

}

/*
void Similarity_Deformations(float* image, int h, int w, float* p, float* q, int Control_Points, float alpha, float density, float* result)
{
    vector<float> gridX(w);
    vector<float> gridY(h);

    float* d_gridX = gridX.data();
    float* d_gridY = gridY.data();

    linspace(0, w, w, d_gridX);
    linspace(0, h, h, d_gridY);

    int row = gridY.size(), column = gridX.size();

    float* Point_X = (float*)malloc(row * column * sizeof(float));
    float* Point_Y = (float*)malloc(row * column * sizeof(float));

    for (int i = 0; i < row * column; ++i)
    {
        int u = i % column;
        int v = i / column;

        Point_X[i] = d_gridX[u];//column
        Point_Y[i] = d_gridY[v];//row
    }

    //d_w => w(i)
    float* d_w = (float*)malloc(Control_Points * row * column * sizeof(float));

    for (int i = 0; i < row * column; ++i)
    {
        for (int j = 0; j < Control_Points; ++j)
        {
            d_w[row * column * j + i] = 1.0 / pow(sqrt((pow(p[2 * j] - Point_X[i], 2) + pow(p[2 * j + 1] - Point_Y[i], 2))), 2 * alpha);
        }
    }

    float* Pstar = (float*)malloc(row * column * 2 * sizeof(float));
    float* Qstar = (float*)malloc(row * column * 2 * sizeof(float));

    //Pstar = p*, Qstar = q*
    for (int i = 0; i < row * column; ++i)
    {
        float w_px = 0, w_py = 0, w_tmp = 0;
        float w_qx = 0, w_qy = 0;

        for (int j = 0; j < Control_Points; ++j)
        {
            w_tmp += d_w[j * row * column + i];
            w_px += d_w[j * row * column + i] * p[2 * j];
            w_py += d_w[j * row * column + i] * p[2 * j + 1];
            w_qx += d_w[j * row * column + i] * q[2 * j];
            w_qy += d_w[j * row * column + i] * q[2 * j + 1];
        }
        Pstar[i] = w_px / w_tmp;
        Pstar[i + row * column] = w_py / w_tmp;
        Qstar[i] = w_qx / w_tmp;
        Qstar[i + row * column] = w_qy / w_tmp;
    }

    float* Phat = (float*)malloc(row * column * 2 * Control_Points * sizeof(float));
    float* Qhat = (float*)malloc(row * column * 2 * Control_Points * sizeof(float));

    //Phat = p^, Qhat = q^
    //p^ = pi - p*, q^ = qi - q*

    for (int i = 0; i < row * column; ++i)
    {
        for (int j = 0; j < Control_Points; ++j)
        {

            Phat[j * row * column * 2 + i] = p[2 * j] - Pstar[i];
            Phat[j * row * column * 2 + row * column + i] = p[2 * j + 1] - Pstar[i + row * column];

            Qhat[j * row * column * 2 + i] = q[2 * j] - Qstar[i];
            Qhat[j * row * column * 2 + row * column + i] = q[2 * j + 1] - Qstar[i + row * column];
            
        }
    }

    float* mu = (float*)malloc(row * column * 2 * sizeof(float));

    for (int i = 0; i < row * column; ++i)
    {
        float tmp1 = 0, tmp2 = 0, tmp3 = 0, tmp4 = 0;

        for (int j = 0; j < Control_Points; ++j)
        {
            
            tmp1 += (Phat[j * row * column * 2 + i] * Phat[j * row * column * 2 + i] * d_w[j * row * column + i]);
            //tmp2 += (Phat[j * row * column * 2 + i] * Phat[j * row * column * 2 + row * column + i] * d_w[j * row * column + i]);
            //tmp3 += (Phat[j * row * column * 2 + row * column + i] * Phat[j * row * column * 2 + i] * d_w[j * row * column + i]);
            tmp4 += (Phat[j * row * column * 2 + row * column + i] * Phat[j * row * column * 2 + row * column + i] * d_w[j * row * column + i]);

        }

        mu[i] = tmp1;
        mu[i + row * column] = tmp4;
    }

    //A(i)
    float* d_a = (float*)malloc(row * column * 4 * Control_Points * sizeof(float));

    for (int i = 0; i < row * column; ++i)
    {
        Matrix2f Matrix1, Transpose_Matrix1;

        Matrix1(0, 0) = Point_X[i] - Pstar[i];
        Matrix1(0, 1) = Point_Y[i] - Pstar[i + row * column];
        Matrix1(1, 0) = Point_Y[i] - Pstar[i + row * column];
        Matrix1(1, 1) = Pstar[i] - Point_X[i];

        Transpose_Matrix1 = Matrix1.transpose();

        for (int j = 0; j < Control_Points; ++j)
        {

            Matrix2f Matrix0, tmp_m;

            // Debug 1
            Matrix0(0, 0) = Phat[j * row * column * 2 + i];
            Matrix0(0, 1) = Phat[j * row * column * 2 + row * column + i];
            Matrix0(1, 0) = Phat[j * row * column * 2 + row * column + i];
            Matrix0(1, 1) = -Phat[j * row * column * 2 + i];
            //Matrix0 << Phat[row * column * 2 + i], Phat[row * column * 2 + row * column + i],
            //    Phat[row * column * 2 + row * column + i], -Phat[row * column * 2 + i];         

            tmp_m = d_w[j * row * column + i] * Matrix0 * Transpose_Matrix1;
            //tmp_m = d_w[j * row * column + i] * Matrix0 * Matrix1;
            
            d_a[j * row * column * 4 + i] = tmp_m(0, 0);
            d_a[j * row * column * 4 + row * column + i] = tmp_m(0, 1);
            d_a[j * row * column * 4 + row * column * 2 + i] = tmp_m(1, 0);
            d_a[j * row * column * 4 + row * column * 3 + i] = tmp_m(1, 1);
            
        }
    }

    float* sum_q_a = (float*)malloc(row * column * 2 * sizeof(float));

    for (int i = 0; i < row * column; ++i)
    {
        float tmp1 = 0, tmp2 = 0;
        for (int j = 0; j < Control_Points; ++j)
        {
            tmp1 += Qhat[j * column * row * 2 + i] * (1.0f / mu[i] * (d_a[j * row * column * 4 + i] + d_a[j * row * column * 4 + row * column * 2 + i]));           
            tmp2 += Qhat[j * column * row * 2 + row * column + i] * (1.0f / mu[i + row * column] * (d_a[j * row * column * 4 + row * column + i] + d_a[j * row * column * 4 + row * column * 3 + i]));

        }
        sum_q_a[i] = tmp1;
        sum_q_a[i + row * column] = tmp2;
    }

    float* transform = (float*)malloc(row * column * 2 * sizeof(float));

    for (int i = 0; i < row * column; ++i)
    {
        transform[2 * i] = sum_q_a[i] + Qstar[i];
        transform[2 * i + 1] = sum_q_a[i + row * column] + Qstar[i + row * column];
    }
    
    for (int i = 0; i < row * column * 2; ++i)
    {
        if (!(transform[i] >= 0) || (transform[i] > h - 1 && i % 2 == 1) || (transform[i] > w - 1 && i % 2 == 0))
        {
            transform[i] = 0;
        }
    }
    
    for (int i = 0; i < row * column; ++i)
    {
        int u = i % column;
        int v = i / column;

        //result(r, g, b) 三通道所以要三個 (transform x, y)
        result[3 * (v * w + u)] = image[3 * (int(transform[2 * i + 1]) * w + int(transform[2 * i]))];
        result[3 * (v * w + u) + 1] = image[3 * (int(transform[2 * i + 1]) * w + int(transform[2 * i])) + 1];
        result[3 * (v * w + u) + 2] = image[3 * (int(transform[2 * i + 1]) * w + int(transform[2 * i])) + 2];
    }
}
*/
//lv(x) = (x− p)M+q
void Similarity_Deformations_Ver2(float* image, int h, int w, float* p, float* q, int Control_Points, float alpha, float density, float* result)
{
    vector<float> gridX(w);
    vector<float> gridY(h);

    float* d_gridX = gridX.data();
    float* d_gridY = gridY.data();

    linspace(0, w, w, d_gridX);
    linspace(0, h, h, d_gridY);

    int row = gridY.size(), column = gridX.size();

    //float* Point_X = (float*)malloc(row * column * sizeof(float));
    //float* Point_Y = (float*)malloc(row * column * sizeof(float));

    float** Point_X = new float* [row];
    float** Point_Y = new float* [row];

    for (int i = 0; i < row; ++i)
    {
        Point_X[i] = new float[column];
        Point_Y[i] = new float[column];
    }

    for (int i = 0; i < row; ++i)
    {
        for (int j = 0; j < column; ++j)
        {
            int u = (i * column + j) % column;
            int v = (i * column + j) / column;

            Point_X[i][j] = d_gridX[u];//column
            Point_Y[i][j] = d_gridY[v];//row
        }
    }

    //d_w => w(i)
    float*** d_w = new float** [Control_Points];

    for (int i = 0; i < Control_Points; ++i)
    {
        d_w[i] = new float* [row];
        for (int j = 0; j < row; ++j)
        {
            d_w[i][j] = new float[column];
        }
    }

    for (int i = 0; i < Control_Points; ++i)
    {
        for (int j = 0; j < row; ++j)
        {
            for (int k = 0; k < column; ++k) {
                d_w[i][j][k] = 1.0 / pow(sqrt((pow(p[2 * i] - Point_X[j][k], 2) + pow(p[2 * i + 1] - Point_Y[j][k], 2))), 2 * alpha);
            }
        }
    }

    //float* Pstar = (float*)malloc(row * column * 2 * sizeof(float));
    //float* Qstar = (float*)malloc(row * column * 2 * sizeof(float));

    //Pstar = p*, Qstar = q*
    float*** Pstar = new float** [2];
    float*** Qstar = new float** [2];

    for (int i = 0; i < 2; ++i)
    {
        Pstar[i] = new float* [row];
        Qstar[i] = new float* [row];
        for (int j = 0; j < row; ++j)
        {
            Pstar[i][j] = new float[column];
            Qstar[i][j] = new float[column];
        }
    }

    //Pstar = p*, Qstar = q*
    for (int i = 0; i < row; ++i)
    {
        for (int j = 0; j < column; ++j) {

            float w_px = 0, w_py = 0, w_tmp = 0;
            float w_qx = 0, w_qy = 0;

            for (int k = 0; k < Control_Points; ++k)
            {
                w_tmp += d_w[k][i][j];
                w_px += d_w[k][i][j] * p[2 * k];
                w_py += d_w[k][i][j] * p[2 * k + 1];
                w_qx += d_w[k][i][j] * q[2 * k];
                w_qy += d_w[k][i][j] * q[2 * k + 1];
            }
            Pstar[0][i][j] = w_px / w_tmp;
            Pstar[1][i][j] = w_py / w_tmp;
            Qstar[0][i][j] = w_qx / w_tmp;
            Qstar[1][i][j] = w_qy / w_tmp;
        }
    }

    //float* Phat = (float*)malloc(row * column * 2 * Control_Points * sizeof(float));
    //float* Qhat = (float*)malloc(row * column * 2 * Control_Points * sizeof(float));

    //Phat = p^, Qhat = q^
    //p^ = pi - p*, q^ = qi - q*

    float**** Phat = new float*** [2];
    float**** Qhat = new float*** [2];

    for (int i = 0; i < 2; ++i)
    {
        Phat[i] = new float** [Control_Points];
        Qhat[i] = new float** [Control_Points];
        for (int j = 0; j < Control_Points; ++j)
        {
            Phat[i][j] = new float* [row];
            Qhat[i][j] = new float* [row];
            for (int k = 0; k < row; ++k)
            {
                Phat[i][j][k] = new float[column];
                Qhat[i][j][k] = new float[column];
            }
        }
    }

    for (int i = 0; i < row; ++i)
    {
        for (int j = 0; j < column; ++j)
        {
            for (int k = 0; k < Control_Points; ++k)
            {
                Phat[0][k][i][j] = p[2 * k] - Pstar[0][i][j];
                Phat[1][k][i][j] = p[2 * k + 1] - Pstar[1][i][j];

                Qhat[0][k][i][j] = q[2 * k] - Qstar[0][i][j];
                Qhat[1][k][i][j] = q[2 * k + 1] - Qstar[1][i][j];
            }
        }
    }

    float** mu = new float* [row];

    for (int i = 0; i < row; ++i)
    {
        mu[i] = new float[column];
    }

    for (int i = 0; i < row; ++i)
    {
        for (int j = 0; j < column; ++j)
        {
            float tmp1 = 0, tmp2 = 0;

            for (int k = 0; k < Control_Points; ++k)
            {
                tmp1 += Phat[0][k][i][j] * Phat[0][k][i][j] * d_w[k][i][j];
                tmp2 += Phat[1][k][i][j] * Phat[1][k][i][j] * d_w[k][i][j];
            }
            mu[i][j] = tmp1 + tmp2;
        }
    }
    
    //float* d_m = (float*)malloc(row * column * 4 * sizeof(float));

    float**** d_m = new float*** [2];

    for (int i = 0; i < 2; ++i)
    {
        d_m[i] = new float** [2];
        for (int j = 0; j < 2; ++j)
        {
            d_m[i][j] = new float* [row];
            for (int k = 0; k < row; ++k)
            {
                d_m[i][j][k] = new float[column];
            }
        }
    }

    for (int i = 0; i < row; ++i)
    {
        for (int j = 0; j < column; ++j)
        {
            float tmp1 = 0, tmp2 = 0, tmp3 = 0, tmp4 = 0;

            for (int k = 0; k < Control_Points; ++k)
            {
                //tmp1 += d_w[j * row * column + i] * (Phat[j * row * column * 2 + i] * Qhat[j * row * column * 2 + i] + Phat[j * row * column * 2 + row * column + i] * Qhat[j * row * column * 2 + row * column + i]);
                //tmp2 += d_w[j * row * column + i] * (Phat[j * row * column * 2 + i] * (-Qhat[j * row * column * 2 + row * column + i]) + Phat[j * row * column * 2 + row * column + i] * Qhat[j * row * column * 2 + i]);
                //tmp3 += d_w[j * row * column + i] * ((-Phat[j * row * column * 2 + row * column + i]) * Qhat[j * row * column * 2 + i] + Phat[j * row * column * 2 + i] * Qhat[j * row * column * 2 + row * column + i]);
                //tmp4 += d_w[j * row * column + i] * (Phat[j * row * column * 2 + row * column + i] * Qhat[j * row * column * 2 + row * column + i] + Phat[j * row * column * 2 + i] * Qhat[j * row * column * 2 + i]);
                tmp1 += d_w[k][i][j] * (Phat[0][k][i][j] * Qhat[0][k][i][j] + Phat[1][k][i][j] * Qhat[1][k][i][j]);
                tmp2 += d_w[k][i][j] * (Phat[0][k][i][j] * Qhat[1][k][i][j] + Phat[1][k][i][j] * -(Qhat[0][k][i][j]));
                tmp3 += d_w[k][i][j] * (Phat[1][k][i][j] * Qhat[0][k][i][j] + (-Phat[0][k][i][j]) * Qhat[1][k][i][j]);
                tmp4 += d_w[k][i][j] * (Phat[1][k][i][j] * Qhat[1][k][i][j] + Phat[0][k][i][j] * Qhat[0][k][i][j]);
            }
            d_m[0][0][i][j] = tmp1 / mu[i][j];
            d_m[0][1][i][j] = tmp2 / mu[i][j];
            d_m[1][0][i][j] = tmp3 / mu[i][j];
            d_m[1][1][i][j] = tmp4 / mu[i][j];
        }
    }

    //float* transform = (float*)malloc(row * column * 2 * sizeof(float));

    float*** transform = new float** [2];

    for (int i = 0; i < 2; ++i)
    {
        transform[i] = new float* [row];
        for (int j = 0; j < row; ++j)
        {
            transform[i][j] = new float[column];
        }
    }

    for (int i = 0; i < row; ++i)
    {
        for (int j = 0; j < column; ++j) {
            transform[0][i][j] = (Point_X[i][j] - Pstar[0][i][j]) * d_m[0][0][i][j] + (Point_Y[i][j] - Pstar[1][i][j]) * d_m[1][0][i][j] + Qstar[0][i][j];
            transform[1][i][j] = (Point_X[i][j] - Pstar[0][i][j]) * d_m[0][1][i][j] + (Point_Y[i][j] - Pstar[1][i][j]) * d_m[1][1][i][j] + Qstar[1][i][j];

            //transform[2 * i] = (Point_X[i] - Qstar[i]) * d_m[4 * i] + (Point_Y[i] - Qstar[i + row * column]) * d_m[4 * i + 2] + Pstar[i];
            //transform[2 * i + 1] = (Point_X[i] - Qstar[i]) * d_m[4 * i + 1] + (Point_Y[i] - Qstar[i + row * column]) * d_m[4 * i + 3] + Pstar[i + row * column];
        }
    }

    for (int i = 0; i < row; ++i)
    {
        for (int j = 0; j < column; ++j)
        {
            for (int k = 0; k < 2; ++k)
            {
                if (!(transform[k][i][j] >= 0) || (transform[k][i][j] > h - 1 && (i * 2 * column + j * 2 + k) % 2 == 1) || (transform[k][i][j] > w - 1 && (i * 2 * column + j * 2 + k) % 2 == 0))
                {
                    transform[k][i][j] = 0;
                }
            }
        }
    }

    for (int i = 0; i < row * column; ++i)
    {
        int u = i % column;
        int v = i / column;

        //result(r, g, b) 三通道所以要三個 (transform x, y)
        result[3 * (v * w + u)] = image[3 * (int(transform[1][v][u]) * w + int(transform[0][v][u]))];
        result[3 * (v * w + u) + 1] = image[3 * (int(transform[1][v][u]) * w + int(transform[0][v][u])) + 1];
        result[3 * (v * w + u) + 2] = image[3 * (int(transform[1][v][u]) * w + int(transform[0][v][u])) + 2];
    }
}

void Similarity_Deformations(float* image, int h, int w, float* p, float* q, int Control_Points, float alpha, float density, float* result)
{
    vector<float> gridX(w);
    vector<float> gridY(h);

    float* d_gridX = gridX.data();
    float* d_gridY = gridY.data();

    linspace(0, w, w, d_gridX);
    linspace(0, h, h, d_gridY);

    int row = gridY.size(), column = gridX.size();

    float* Point_X = (float*)malloc(row * column * sizeof(float));
    float* Point_Y = (float*)malloc(row * column * sizeof(float));

    for (int i = 0; i < row * column; ++i)
    {
        int u = i % column;
        int v = i / column;

        Point_X[i] = d_gridX[u];//column
        Point_Y[i] = d_gridY[v];//row
    }

    //d_w => w(i)
    float* d_w = (float*)malloc(Control_Points * row * column * sizeof(float));

    for (int i = 0; i < row * column; ++i)
    {
        for (int j = 0; j < Control_Points; ++j)
        {
            d_w[row * column * j + i] = 1.0 / pow(sqrt((pow(p[2 * j] - Point_X[i], 2) + pow(p[2 * j + 1] - Point_Y[i], 2))), 2 * alpha);
        }
    }

    float* Pstar = (float*)malloc(row * column * 2 * sizeof(float));
    float* Qstar = (float*)malloc(row * column * 2 * sizeof(float));

    //Pstar = p*, Qstar = q*
    for (int i = 0; i < row * column; ++i)
    {
        float w_px = 0, w_py = 0, w_tmp = 0;
        float w_qx = 0, w_qy = 0;

        for (int j = 0; j < Control_Points; ++j)
        {
            w_tmp += d_w[j * row * column + i];
            w_px += d_w[j * row * column + i] * p[2 * j];
            w_py += d_w[j * row * column + i] * p[2 * j + 1];
            w_qx += d_w[j * row * column + i] * q[2 * j];
            w_qy += d_w[j * row * column + i] * q[2 * j + 1];
        }
        Pstar[i] = w_px / w_tmp;
        Pstar[i + row * column] = w_py / w_tmp;
        Qstar[i] = w_qx / w_tmp;
        Qstar[i + row * column] = w_qy / w_tmp;
    }

    float* Phat = (float*)malloc(row * column * 2 * Control_Points * sizeof(float));
    float* Qhat = (float*)malloc(row * column * 2 * Control_Points * sizeof(float));

    //Phat = p^, Qhat = q^
    //p^ = pi - p*, q^ = qi - q*

    for (int i = 0; i < row * column; ++i)
    {
        for (int j = 0; j < Control_Points; ++j)
        {

            Phat[j * row * column * 2 + i] = p[2 * j] - Pstar[i];
            Phat[j * row * column * 2 + row * column + i] = p[2 * j + 1] - Pstar[i + row * column];

            Qhat[j * row * column * 2 + i] = q[2 * j] - Qstar[i];
            Qhat[j * row * column * 2 + row * column + i] = q[2 * j + 1] - Qstar[i + row * column];

        }
    }

    float* mu = (float*)malloc(row * column * sizeof(float));

    for (int i = 0; i < row * column; ++i)
    {
        float tmp1 = 0, tmp2 = 0;

        for (int j = 0; j < Control_Points; ++j)
        {
            tmp1 += (Phat[j * row * column * 2 + i] * Phat[j * row * column * 2 + i] * d_w[j * row * column + i]);
            tmp2 += (Phat[j * row * column * 2 + row * column + i] * Phat[j * row * column * 2 + row * column + i] * d_w[j * row * column + i]);
        }
        mu[i] = tmp1 + tmp2;
    }

    float* d_m = (float*)malloc(row * column * 4 * sizeof(float));

    for (int i = 0; i < row * column; ++i)
    {
        float tmp1 = 0, tmp2 = 0, tmp3 = 0, tmp4 = 0;

        for (int j = 0; j < Control_Points; ++j)
        {
            //tmp1 += d_w[j * row * column + i] * (Phat[j * row * column * 2 + i] * Qhat[j * row * column * 2 + i] + Phat[j * row * column * 2 + row * column + i] * Qhat[j * row * column * 2 + row * column + i]);
            //tmp2 += d_w[j * row * column + i] * (Phat[j * row * column * 2 + i] * (-Qhat[j * row * column * 2 + row * column + i]) + Phat[j * row * column * 2 + row * column + i] * Qhat[j * row * column * 2 + i]);
            //tmp3 += d_w[j * row * column + i] * ((-Phat[j * row * column * 2 + row * column + i]) * Qhat[j * row * column * 2 + i] + Phat[j * row * column * 2 + i] * Qhat[j * row * column * 2 + row * column + i]);
            //tmp4 += d_w[j * row * column + i] * (Phat[j * row * column * 2 + row * column + i] * Qhat[j * row * column * 2 + row * column + i] + Phat[j * row * column * 2 + i] * Qhat[j * row * column * 2 + i]);
            tmp1 += d_w[j * row * column + i] * (Phat[j * row * column * 2 + i] * Qhat[j * row * column * 2 + i] + Phat[j * row * column * 2 + row * column + i] * Qhat[j * row * column * 2 + row * column + i]);
            tmp2 += d_w[j * row * column + i] * (Phat[j * row * column * 2 + i] * Qhat[j * row * column * 2 + row * column + i] + Phat[j * row * column * 2 + row * column + i] * -(Qhat[j * row * column * 2 + i]));
            tmp3 += d_w[j * row * column + i] * (Phat[j * row * column * 2 + row * column + i] * Qhat[j * row * column * 2 + i] + (-Phat[j * row * column * 2 + i]) * Qhat[j * row * column * 2 + row * column + i]);
            tmp4 += d_w[j * row * column + i] * (Phat[j * row * column * 2 + row * column + i] * Qhat[j * row * column * 2 + row * column + i] + Phat[j * row * column * 2 + i] * Qhat[j * row * column * 2 + i]);
        }
        d_m[4 * i] = tmp1 / mu[i];
        d_m[4 * i + 1] = tmp2 / mu[i];
        d_m[4 * i + 2] = tmp3 / mu[i];
        d_m[4 * i + 3] = tmp4 / mu[i];
    }

    float* transform = (float*)malloc(row * column * 2 * sizeof(float));

    for (int i = 0; i < row * column; ++i)
    {
        transform[2 * i] = (Point_X[i] - Pstar[i]) * d_m[4 * i] + (Point_Y[i] - Pstar[i + row * column]) * d_m[4 * i + 2] + Qstar[i];
        transform[2 * i + 1] = (Point_X[i] - Pstar[i]) * d_m[4 * i + 1] + (Point_Y[i] - Pstar[i + row * column]) * d_m[4 * i + 3] + Qstar[i + row * column];
    }

    for (int i = 0; i < row * column * 2; ++i)
    {
        if (!(transform[i] >= 0) || (transform[i] > h - 1 && i % 2 == 1) || (transform[i] > w - 1 && i % 2 == 0))
        {
            transform[i] = 0;
        }
    }

    for (int i = 0; i < row * column; ++i)
    {
        int u = i % column;
        int v = i / column;

        //result(r, g, b) 三通道所以要三個 (transform x, y)
        result[3 * (v * w + u)] = image[3 * (int(transform[2 * i + 1]) * w + int(transform[2 * i]))];
        result[3 * (v * w + u) + 1] = image[3 * (int(transform[2 * i + 1]) * w + int(transform[2 * i])) + 1];
        result[3 * (v * w + u) + 2] = image[3 * (int(transform[2 * i + 1]) * w + int(transform[2 * i])) + 2];
    }
}

/*
// h = 352, w = 409, p.size() = 14, q.size() = 14, Control_Points = 7, alpha = 1.0, density = 1.0
void Rigid_Deformations(float* image, int h, int w, float* p, float* q, int Control_Points, float alpha, float density, float* result)
{
    vector<float> gridX(w);
    vector<float> gridY(h);

    float* d_gridX = gridX.data();
    float* d_gridY = gridY.data();

    linspace(0, w, w, d_gridX);
    linspace(0, h, h, d_gridY);

    int row = gridY.size(), column = gridX.size();

    float* Point_X = (float*)malloc(row * column * sizeof(float));
    float* Point_Y = (float*)malloc(row * column * sizeof(float));

    for (int i = 0; i < row * column; ++i)
    {
        int u = i % column;
        int v = i / column;

        Point_X[i] = d_gridX[u];//column
        Point_Y[i] = d_gridY[v];//row
        // cout << d_gridX[u] << " " << d_gridY[v] << endl;
        // 175 * 205
    }

    //d_w => w(i)
    float* d_w = (float*)malloc(Control_Points * row * column * sizeof(float));

    for (int i = 0; i < row * column; ++i)
    {
        for (int j = 0; j < Control_Points; ++j)
        {
            d_w[row * column * j + i] = 1.0 / pow(sqrt((pow(p[2 * j] - Point_X[i], 2) + pow(p[2 * j + 1] - Point_Y[i], 2))), 2 * alpha);
        }
    }

    float* Pstar = (float*)malloc(row * column * 2 * sizeof(float));
    float* Qstar = (float*)malloc(row * column * 2 * sizeof(float));

    //Pstar = p*, Qstar = q*
    for (int i = 0; i < row * column; ++i)
    {
        float w_px = 0, w_py = 0, w_tmp = 0;
        float w_qx = 0, w_qy = 0;

        for (int j = 0; j < Control_Points; ++j)
        {
            w_tmp += d_w[j * row * column + i];
            w_px += d_w[j * row * column + i] * p[2 * j];
            w_py += d_w[j * row * column + i] * p[2 * j + 1];
            w_qx += d_w[j * row * column + i] * q[2 * j];
            w_qy += d_w[j * row * column + i] * q[2 * j + 1];
        }
        Pstar[i] = w_px / w_tmp;
        Pstar[i + row * column] = w_py / w_tmp;
        Qstar[i] = w_qx / w_tmp;
        Qstar[i + row * column] = w_qy / w_tmp;
    }

    float* Phat = (float*)malloc(row * column * 2 * Control_Points * sizeof(float));
    float* Qhat = (float*)malloc(row * column * 2 * Control_Points * sizeof(float));

    //Phat = p^, Qhat = q^
    //p^ = pi - p*, q^ = qi - q*

    for (int i = 0; i < row * column; ++i)
    {
        for (int j = 0; j < Control_Points; ++j)
        {

            Phat[j * row * column * 2 + i] = p[2 * j] - Pstar[i];
            Phat[j * row * column * 2 + row * column + i] = p[2 * j + 1] - Pstar[i + row * column];

            Qhat[j * row * column * 2 + i] = q[2 * j] - Qstar[i];
            Qhat[j * row * column * 2 + row * column + i] = q[2 * j + 1] - Qstar[i + row * column];

        }
    }

    float* mu = (float*)malloc(row * column * 2 * sizeof(float));

    for (int i = 0; i < row * column; ++i)
    {
        float tmp1 = 0, tmp2 = 0, tmp3 = 0, tmp4 = 0;

        for (int j = 0; j < Control_Points; ++j)
        {

            tmp1 += (Qhat[j * row * column * 2 + i] * Phat[j * row * column * 2 + i] * d_w[j * row * column + i]);
            tmp2 += (Qhat[j * row * column * 2 + row * column + i] * Phat[j * row * column * 2 + row * column + i] * d_w[j * row * column + i]);
            tmp3 += (Qhat[j * row * column * 2 + i] * -Phat[j * row * column * 2 + row * column + i] * d_w[j * row * column + i]);
            tmp4 += (Qhat[j * row * column * 2 + row * column + i] * Phat[j * row * column * 2 + i] * d_w[j * row * column + i]);

        }

        mu[i] = sqrt(tmp1 * tmp1 + tmp3 * tmp3);
        mu[i + row * column] = sqrt(tmp2 * tmp2 + tmp4 * tmp4);
    }

    //A(i)
    float* d_a = (float*)malloc(row * column * 4 * Control_Points * sizeof(float));

    for (int i = 0; i < row * column; ++i)
    {
        Matrix2f Matrix1, Transpose_Matrix1;

        Matrix1(0, 0) = Point_X[i] - Pstar[i];
        Matrix1(0, 1) = Point_Y[i] - Pstar[i + row * column];
        Matrix1(1, 0) = Point_Y[i] - Pstar[i + row * column];
        Matrix1(1, 1) = Pstar[i] - Point_X[i];

        Transpose_Matrix1 = Matrix1.transpose();

        for (int j = 0; j < Control_Points; ++j)
        {

            Matrix2f Matrix0, tmp_m;

            // Debug 1
            Matrix0(0, 0) = Phat[j * row * column * 2 + i];
            Matrix0(0, 1) = Phat[j * row * column * 2 + row * column + i];
            Matrix0(1, 0) = Phat[j * row * column * 2 + row * column + i];
            Matrix0(1, 1) = -Phat[j * row * column * 2 + i];
            //Matrix0 << Phat[row * column * 2 + i], Phat[row * column * 2 + row * column + i],
            //    Phat[row * column * 2 + row * column + i], -Phat[row * column * 2 + i];         

            tmp_m = d_w[j * row * column + i] * Matrix0 * Transpose_Matrix1;
            //tmp_m = d_w[j * row * column + i] * Matrix0 * Matrix1;

            d_a[j * row * column * 4 + i] = tmp_m(0, 0);
            d_a[j * row * column * 4 + row * column + i] = tmp_m(0, 1);
            d_a[j * row * column * 4 + row * column * 2 + i] = tmp_m(1, 0);
            d_a[j * row * column * 4 + row * column * 3 + i] = tmp_m(1, 1);

        }
    }

    float* sum_q_a = (float*)malloc(row * column * 2 * sizeof(float));

    for (int i = 0; i < row * column; ++i)
    {
        float tmp1 = 0, tmp2 = 0;
        for (int j = 0; j < Control_Points; ++j)
        {
            tmp1 += Qhat[j * column * row * 2 + i] * (d_a[j * row * column * 4 + i] + d_a[j * row * column * 4 + row * column * 2 + i]) / mu[i];
            tmp2 += Qhat[j * column * row * 2 + row * column + i] * (d_a[j * row * column * 4 + row * column + i] + d_a[j * row * column * 4 + row * column * 3 + i]) / mu[i + row * column];

        }
        sum_q_a[i] = tmp1;
        sum_q_a[i + row * column] = tmp2;
    }

    float* transform = (float*)malloc(row * column * 2 * sizeof(float));

    for (int i = 0; i < row * column; ++i)
    {
        transform[2 * i] = ((Point_X[i] - Pstar[i]) / sqrt(pow(Point_X[i] - Pstar[i], 2) + pow(Point_Y[i] - Pstar[i + row * column], 2)))
            * (sum_q_a[i] / sqrt(pow(sum_q_a[i], 2) + pow(sum_q_a[i + row * column], 2))) + Qstar[i];
        transform[2 * i + 1] = ((Point_Y[i] - Pstar[i + row * column]) / sqrt(pow(Point_X[i] - Pstar[i], 2) + pow(Point_Y[i] - Pstar[i + row * column], 2)))
            * (sum_q_a[i + row * column] / sqrt(pow(sum_q_a[i], 2) + pow(sum_q_a[i + row * column], 2))) + Qstar[i + row * column];
    }

    for (int i = 0; i < row * column * 2; ++i)
    {
        if (!(transform[i] >= 0) || (transform[i] > h - 1 && i % 2 == 1) || (transform[i] > w - 1 && i % 2 == 0))
        {
            transform[i] = 0;
        }
    }

    for (int i = 0; i < row * column; ++i)
    {
        int u = i % column;
        int v = i / column;

        //result[3 * (int(transform[2 * i + 1]) * w + int(transform[2 * i]))] = image[3 * (v * w + u)]; 
        //result[3 * (int(transform[2 * i + 1]) * w + int(transform[2 * i])) + 1] = image[3 * (v * w + u) + 1]; 
        //result[3 * (int(transform[2 * i + 1]) * w + int(transform[2 * i])) + 2] = image[3 * (v * w + u) + 2]; 

        //result(r, g, b) 三通道所以要三個 (transform x, y)
        result[3 * (v * w + u)] = image[3 * (int(transform[2 * i + 1]) * w + int(transform[2 * i]))];
        result[3 * (v * w + u) + 1] = image[3 * (int(transform[2 * i + 1]) * w + int(transform[2 * i])) + 1];
        result[3 * (v * w + u) + 2] = image[3 * (int(transform[2 * i + 1]) * w + int(transform[2 * i])) + 2];
    }

}
*/

void Rigid_Deformations(float* image, int h, int w, float* p, float* q, int Control_Points, float alpha, float density, float* result)
{
    vector<float> gridX(w);
    vector<float> gridY(h);

    float* d_gridX = gridX.data();
    float* d_gridY = gridY.data();

    linspace(0, w, w, d_gridX);
    linspace(0, h, h, d_gridY);

    int row = gridY.size(), column = gridX.size();

    float* Point_X = (float*)malloc(row * column * sizeof(float));
    float* Point_Y = (float*)malloc(row * column * sizeof(float));

    for (int i = 0; i < row * column; ++i)
    {
        int u = i % column;
        int v = i / column;

        Point_X[i] = d_gridX[u];//column
        Point_Y[i] = d_gridY[v];//row
    }

    //d_w => w(i)
    float* d_w = (float*)malloc(Control_Points * row * column * sizeof(float));

    for (int i = 0; i < row * column; ++i)
    {
        for (int j = 0; j < Control_Points; ++j)
        {
            d_w[row * column * j + i] = 1.0 / pow(sqrt((pow(p[2 * j] - Point_X[i], 2) + pow(p[2 * j + 1] - Point_Y[i], 2))), 2 * alpha);
        }
    }

    float* Pstar = (float*)malloc(row * column * 2 * sizeof(float));
    float* Qstar = (float*)malloc(row * column * 2 * sizeof(float));

    //Pstar = p*, Qstar = q*
    for (int i = 0; i < row * column; ++i)
    {
        float w_px = 0, w_py = 0, w_tmp = 0;
        float w_qx = 0, w_qy = 0;

        for (int j = 0; j < Control_Points; ++j)
        {
            w_tmp += d_w[j * row * column + i];
            w_px += d_w[j * row * column + i] * p[2 * j];
            w_py += d_w[j * row * column + i] * p[2 * j + 1];
            w_qx += d_w[j * row * column + i] * q[2 * j];
            w_qy += d_w[j * row * column + i] * q[2 * j + 1];
        }
        Pstar[i] = w_px / w_tmp;
        Pstar[i + row * column] = w_py / w_tmp;
        Qstar[i] = w_qx / w_tmp;
        Qstar[i + row * column] = w_qy / w_tmp;
    }

    float* Phat = (float*)malloc(row * column * 2 * Control_Points * sizeof(float));
    float* Qhat = (float*)malloc(row * column * 2 * Control_Points * sizeof(float));

    //Phat = p^, Qhat = q^
    //p^ = pi - p*, q^ = qi - q*

    for (int i = 0; i < row * column; ++i)
    {
        for (int j = 0; j < Control_Points; ++j)
        {

            Phat[j * row * column * 2 + i] = p[2 * j] - Pstar[i];
            Phat[j * row * column * 2 + row * column + i] = p[2 * j + 1] - Pstar[i + row * column];

            Qhat[j * row * column * 2 + i] = q[2 * j] - Qstar[i];
            Qhat[j * row * column * 2 + row * column + i] = q[2 * j + 1] - Qstar[i + row * column];

        }
    }

    float* mu = (float*)malloc(row * column * sizeof(float));

    for (int i = 0; i < row * column; ++i)
    {
        float tmp1 = 0, tmp2 = 0, tmp3 = 0, tmp4 = 0;

        for (int j = 0; j < Control_Points; ++j)
        {

            tmp1 += (Qhat[j * row * column * 2 + i] * Phat[j * row * column * 2 + i] * d_w[j * row * column + i]);
            tmp2 += (Qhat[j * row * column * 2 + row * column + i] * Phat[j * row * column * 2 + row * column + i] * d_w[j * row * column + i]);
            tmp3 += (Qhat[j * row * column * 2 + i] * -Phat[j * row * column * 2 + row * column + i] * d_w[j * row * column + i]);
            tmp4 += (Qhat[j * row * column * 2 + row * column + i] * Phat[j * row * column * 2 + i] * d_w[j * row * column + i]);

        }
        //mu[i] = sqrt(tmp1 * tmp1 + tmp2 * tmp2) + sqrt(tmp3 * tmp3 + tmp4 * tmp4);       
        //mu[i] = sqrt(tmp1 * tmp1 + tmp2 * tmp2 + tmp3 * tmp3 + tmp4 * tmp4);  
        mu[i] = sqrt(pow(tmp1 + tmp2, 2) + pow(tmp3 + tmp4, 2));
    }

    float* d_m = (float*)malloc(row * column * 4 * sizeof(float));

    for (int i = 0; i < row * column; ++i)
    {
        float tmp1 = 0, tmp2 = 0, tmp3 = 0, tmp4 = 0;

        for (int j = 0; j < Control_Points; ++j)
        {
            //tmp1 += d_w[j * row * column + i] * (Phat[j * row * column * 2 + i] * Qhat[j * row * column * 2 + i] + Phat[j * row * column * 2 + row * column + i] * Qhat[j * row * column * 2 + row * column + i]);
            //tmp2 += d_w[j * row * column + i] * (Phat[j * row * column * 2 + i] * (-Qhat[j * row * column * 2 + row * column + i]) + Phat[j * row * column * 2 + row * column + i] * Qhat[j * row * column * 2 + i]);
            //tmp3 += d_w[j * row * column + i] * ((-Phat[j * row * column * 2 + row * column + i]) * Qhat[j * row * column * 2 + i] + Phat[j * row * column * 2 + i] * Qhat[j * row * column * 2 + row * column + i]);
            //tmp4 += d_w[j * row * column + i] * (Phat[j * row * column * 2 + row * column + i] * Qhat[j * row * column * 2 + row * column + i] + Phat[j * row * column * 2 + i] * Qhat[j * row * column * 2 + i]);
            tmp1 += d_w[j * row * column + i] * (Phat[j * row * column * 2 + i] * Qhat[j * row * column * 2 + i] + Phat[j * row * column * 2 + row * column + i] * Qhat[j * row * column * 2 + row * column + i]);
            tmp2 += d_w[j * row * column + i] * (Phat[j * row * column * 2 + i] * Qhat[j * row * column * 2 + row * column + i] + Phat[j * row * column * 2 + row * column + i] * -(Qhat[j * row * column * 2 + i]));
            tmp3 += d_w[j * row * column + i] * (Phat[j * row * column * 2 + row * column + i] * Qhat[j * row * column * 2 + i] + (-Phat[j * row * column * 2 + i]) * Qhat[j * row * column * 2 + row * column + i]);
            tmp4 += d_w[j * row * column + i] * (Phat[j * row * column * 2 + row * column + i] * Qhat[j * row * column * 2 + row * column + i] + Phat[j * row * column * 2 + i] * Qhat[j * row * column * 2 + i]);
        }
        d_m[4 * i] = tmp1 / mu[i];
        d_m[4 * i + 1] = tmp2 / mu[i];
        d_m[4 * i + 2] = tmp3 / mu[i];
        d_m[4 * i + 3] = tmp4 / mu[i];
    }

    float* transform = (float*)malloc(row * column * 2 * sizeof(float));

    for (int i = 0; i < row * column; ++i)
    {
        transform[2 * i] = (Point_X[i] - Pstar[i]) * d_m[4 * i] + (Point_Y[i] - Pstar[i + row * column]) * d_m[4 * i + 2] + Qstar[i];
        transform[2 * i + 1] = (Point_X[i] - Pstar[i]) * d_m[4 * i + 1] + (Point_Y[i] - Pstar[i + row * column]) * d_m[4 * i + 3] + Qstar[i + row * column];
    }

    for (int i = 0; i < row * column * 2; ++i)
    {
        if (!(transform[i] >= 0) || (transform[i] > h - 1 && i % 2 == 1) || (transform[i] > w - 1 && i % 2 == 0))
        {
            transform[i] = 0;
        }
    }

    for (int i = 0; i < row * column; ++i)
    {
        int u = i % column;
        int v = i / column;

        //result(r, g, b) 三通道所以要三個 (transform x, y)
        result[3 * (v * w + u)] = image[3 * (int(transform[2 * i + 1]) * w + int(transform[2 * i]))];
        result[3 * (v * w + u) + 1] = image[3 * (int(transform[2 * i + 1]) * w + int(transform[2 * i])) + 1];
        result[3 * (v * w + u) + 2] = image[3 * (int(transform[2 * i + 1]) * w + int(transform[2 * i])) + 2];
    }
}

void Rigid_Deformations_Ver2(float* image, int h, int w, float* p, float* q, int Control_Points, float alpha, float density, float* result)
{
    vector<float> gridX(w);
    vector<float> gridY(h);

    float* d_gridX = gridX.data();
    float* d_gridY = gridY.data();

    linspace(0, w, w, d_gridX);
    linspace(0, h, h, d_gridY);

    int row = gridY.size(), column = gridX.size();

    float** Point_X = new float* [row];
    float** Point_Y = new float* [row];

    for (int i = 0; i < row; ++i)
    {
        Point_X[i] = new float[column];
        Point_Y[i] = new float[column];
    }

    for (int i = 0; i < row; ++i)
    {
        for (int j = 0; j < column; ++j)
        {
            int u = (i * column + j) % column;
            int v = (i * column + j) / column;

            Point_X[i][j] = d_gridX[u];//column
            Point_Y[i][j] = d_gridY[v];//row
        }
    }

    //d_w => w(i)
    float*** d_w = new float** [Control_Points];

    for (int i = 0; i < Control_Points; ++i)
    {
        d_w[i] = new float* [row];
        for (int j = 0; j < row; ++j)
        {
            d_w[i][j] = new float[column];
        }
    }

    for (int i = 0; i < Control_Points; ++i)
    {
        for (int j = 0; j < row; ++j)
        {
            for (int k = 0; k < column; ++k) {
                d_w[i][j][k] = 1.0 / pow(sqrt((pow(p[2 * i] - Point_X[j][k], 2) + pow(p[2 * i + 1] - Point_Y[j][k], 2))), 2 * alpha);
            }
        }
    }

    //float* Pstar = (float*)malloc(row * column * 2 * sizeof(float));
    //float* Qstar = (float*)malloc(row * column * 2 * sizeof(float));

    //Pstar = p*, Qstar = q*
    float*** Pstar = new float** [2];
    float*** Qstar = new float** [2];

    for (int i = 0; i < 2; ++i)
    {
        Pstar[i] = new float* [row];
        Qstar[i] = new float* [row];
        for (int j = 0; j < row; ++j)
        {
            Pstar[i][j] = new float[column];
            Qstar[i][j] = new float[column];
        }
    }

    //Pstar = p*, Qstar = q*
    for (int i = 0; i < row; ++i)
    {
        for (int j = 0; j < column; ++j) {

            float w_px = 0, w_py = 0, w_tmp = 0;
            float w_qx = 0, w_qy = 0;

            for (int k = 0; k < Control_Points; ++k)
            {
                w_tmp += d_w[k][i][j];
                w_px += d_w[k][i][j] * p[2 * k];
                w_py += d_w[k][i][j] * p[2 * k + 1];
                w_qx += d_w[k][i][j] * q[2 * k];
                w_qy += d_w[k][i][j] * q[2 * k + 1];
            }
            Pstar[0][i][j] = w_px / w_tmp;
            Pstar[1][i][j] = w_py / w_tmp;
            Qstar[0][i][j] = w_qx / w_tmp;
            Qstar[1][i][j] = w_qy / w_tmp;
        }
    }

    //float* Phat = (float*)malloc(row * column * 2 * Control_Points * sizeof(float));
    //float* Qhat = (float*)malloc(row * column * 2 * Control_Points * sizeof(float));

    //Phat = p^, Qhat = q^
    //p^ = pi - p*, q^ = qi - q*

    float**** Phat = new float*** [2];
    float**** Qhat = new float*** [2];

    for (int i = 0; i < 2; ++i)
    {
        Phat[i] = new float** [Control_Points];
        Qhat[i] = new float** [Control_Points];
        for (int j = 0; j < Control_Points; ++j)
        {
            Phat[i][j] = new float* [row];
            Qhat[i][j] = new float* [row];
            for (int k = 0; k < row; ++k)
            {
                Phat[i][j][k] = new float[column];
                Qhat[i][j][k] = new float[column];
            }
        }
    }

    for (int i = 0; i < row; ++i)
    {
        for (int j = 0; j < column; ++j)
        {
            for (int k = 0; k < Control_Points; ++k)
            {
                Phat[0][k][i][j] = p[2 * k] - Pstar[0][i][j];
                Phat[1][k][i][j] = p[2 * k + 1] - Pstar[1][i][j];

                Qhat[0][k][i][j] = q[2 * k] - Qstar[0][i][j];
                Qhat[1][k][i][j] = q[2 * k + 1] - Qstar[1][i][j];
            }
        }
    }

    float** mu = new float* [row];

    for (int i = 0; i < row; ++i)
    {
        mu[i] = new float[column];
    }

    for (int i = 0; i < row; ++i)
    {
        for (int j = 0; j < column; ++j) {

            float tmp1 = 0, tmp2 = 0, tmp3 = 0, tmp4 = 0;

            for (int k = 0; k < Control_Points; ++k)
            {

                tmp1 += Qhat[0][k][i][j] * Phat[0][k][i][j] * d_w[k][i][j];
                tmp2 += Qhat[1][k][i][j] * Phat[1][k][i][j] * d_w[k][i][j];
                tmp3 += Qhat[0][k][i][j] * -Phat[1][k][i][j] * d_w[k][i][j];
                tmp4 += Qhat[1][k][i][j] * Phat[0][k][i][j] * d_w[k][i][j];

            }
            //mu[i] = sqrt(tmp1 * tmp1 + tmp2 * tmp2) + sqrt(tmp3 * tmp3 + tmp4 * tmp4);       
            //mu[i] = sqrt(tmp1 * tmp1 + tmp2 * tmp2 + tmp3 * tmp3 + tmp4 * tmp4);  
            mu[i][j] = sqrt(pow(tmp1 + tmp2, 2) + pow(tmp3 + tmp4, 2));
        }
    }

    //float* d_m = (float*)malloc(row * column * 4 * sizeof(float));

    float**** d_m = new float*** [2];

    for (int i = 0; i < 2; ++i)
    {
        d_m[i] = new float** [2];
        for (int j = 0; j < 2; ++j)
        {
            d_m[i][j] = new float* [row];
            for (int k = 0; k < row; ++k)
            {
                d_m[i][j][k] = new float[column];
            }
        }
    }

    for (int i = 0; i < row; ++i)
    {
        for (int j = 0; j < column; ++j) {

            float tmp1 = 0, tmp2 = 0, tmp3 = 0, tmp4 = 0;

            for (int k = 0; k < Control_Points; ++k)
            {
                //tmp1 += d_w[j * row * column + i] * (Phat[j * row * column * 2 + i] * Qhat[j * row * column * 2 + i] + Phat[j * row * column * 2 + row * column + i] * Qhat[j * row * column * 2 + row * column + i]);
                //tmp2 += d_w[j * row * column + i] * (Phat[j * row * column * 2 + i] * (-Qhat[j * row * column * 2 + row * column + i]) + Phat[j * row * column * 2 + row * column + i] * Qhat[j * row * column * 2 + i]);
                //tmp3 += d_w[j * row * column + i] * ((-Phat[j * row * column * 2 + row * column + i]) * Qhat[j * row * column * 2 + i] + Phat[j * row * column * 2 + i] * Qhat[j * row * column * 2 + row * column + i]);
                //tmp4 += d_w[j * row * column + i] * (Phat[j * row * column * 2 + row * column + i] * Qhat[j * row * column * 2 + row * column + i] + Phat[j * row * column * 2 + i] * Qhat[j * row * column * 2 + i]);
                tmp1 += d_w[k][i][j] * (Phat[0][k][i][j] * Qhat[0][k][i][j] + Phat[1][k][i][j] * Qhat[1][k][i][j]);
                tmp2 += d_w[k][i][j] * (Phat[0][k][i][j] * Qhat[1][k][i][j] + Phat[1][k][i][j] * -(Qhat[0][k][i][j]));
                tmp3 += d_w[k][i][j] * (Phat[1][k][i][j] * Qhat[0][k][i][j] + (-Phat[0][k][i][j]) * Qhat[1][k][i][j]);
                tmp4 += d_w[k][i][j] * (Phat[1][k][i][j] * Qhat[1][k][i][j] + Phat[0][k][i][j] * Qhat[0][k][i][j]);
            }
            d_m[0][0][i][j] = tmp1 / mu[i][j];
            d_m[0][1][i][j] = tmp2 / mu[i][j];
            d_m[1][0][i][j] = tmp3 / mu[i][j];
            d_m[1][1][i][j] = tmp4 / mu[i][j];
        }
    }

    float*** transform = new float** [2];

    for (int i = 0; i < 2; ++i)
    {
        transform[i] = new float* [row];
        for (int j = 0; j < row; ++j)
        {
            transform[i][j] = new float[column];
        }
    }

    for (int i = 0; i < row; ++i)
    {
        for (int j = 0; j < column; ++j) {
            transform[0][i][j] = (Point_X[i][j] - Pstar[0][i][j]) * d_m[0][0][i][j] + (Point_Y[i][j] - Pstar[1][i][j]) * d_m[1][0][i][j] + Qstar[0][i][j];
            transform[1][i][j] = (Point_X[i][j] - Pstar[0][i][j]) * d_m[0][1][i][j] + (Point_Y[i][j] - Pstar[1][i][j]) * d_m[1][1][i][j] + Qstar[1][i][j];

            //transform[2 * i] = (Point_X[i] - Qstar[i]) * d_m[4 * i] + (Point_Y[i] - Qstar[i + row * column]) * d_m[4 * i + 2] + Pstar[i];
            //transform[2 * i + 1] = (Point_X[i] - Qstar[i]) * d_m[4 * i + 1] + (Point_Y[i] - Qstar[i + row * column]) * d_m[4 * i + 3] + Pstar[i + row * column];
        }
    }

    for (int i = 0; i < row; ++i)
    {
        for (int j = 0; j < column; ++j)
        {
            for (int k = 0; k < 2; ++k)
            {
                if (!(transform[k][i][j] >= 0) || (transform[k][i][j] > h - 1 && (i * 2 * column + j * 2 + k) % 2 == 1) || (transform[k][i][j] > w - 1 && (i * 2 * column + j * 2 + k) % 2 == 0))
                {
                    transform[k][i][j] = 0;
                }
            }
        }
    }

    for (int i = 0; i < row * column; ++i)
    {
        int u = i % column;
        int v = i / column;

        //result(r, g, b) 三通道所以要三個 (transform x, y)
        result[3 * (v * w + u)] = image[3 * (int(transform[1][v][u]) * w + int(transform[0][v][u]))];
        result[3 * (v * w + u) + 1] = image[3 * (int(transform[1][v][u]) * w + int(transform[0][v][u])) + 1];
        result[3 * (v * w + u) + 2] = image[3 * (int(transform[1][v][u]) * w + int(transform[0][v][u])) + 2];
    }
}