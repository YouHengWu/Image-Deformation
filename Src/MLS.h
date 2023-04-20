#ifndef __MLS_H__
#define __MLS_H__
#include <vector>
#include <string>

void Affine_Deformations(float *image, int h, int w, float *p, float *q, int Control_Points, float alpha, float density, float *result);

void Similarity_Deformations(float* image, int h, int w, float* p, float* q, int Control_Points, float alpha, float density, float* result);

void Rigid_Deformations(float* image, int h, int w, float* p, float* q, int Control_Points, float alpha, float density, float* result);

void Affine_Deformations_Ver2(float* image, int h, int w, float* p, float* q, int Control_Points, float alpha, float density, float* result);

void Similarity_Deformations_Ver2(float* image, int h, int w, float* p, float* q, int Control_Points, float alpha, float density, float* result);

void Rigid_Deformations_Ver2(float* image, int h, int w, float* p, float* q, int Control_Points, float alpha, float density, float* result);

#endif
