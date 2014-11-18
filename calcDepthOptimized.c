// CS 61C Fall 2014 Project 3

#include <limits.h>
#include <math.h>
#include <stdbool.h>
#include <stdio.h>
#include "utils.h"

// include SSE intrinsics
#if defined(_MSC_VER)
#include <intrin.h>
#elif defined(__GNUC__) && (defined(__x86_64__) || defined(__i386__))
#include <x86intrin.h>
#endif

#include "calcDepthOptimized.h"
#include "calcDepthNaive.h"
#include <memory.h>

#define ABS(x) (((x) < 0) ? (-(x)) : (x))

float displacementNaiveMemo(int dx, int dy, int *pos, float known[][3])
{
	if(*pos < 9999){
		for(int i = 0; i < *pos; i++)
			if((known[i][0] == dx) && (known[i][1] == dy))
				return known[i][2];
	}
	float squaredDisplacement = dx * dx + dy * dy;
	float displacement = sqrt(squaredDisplacement);
	if(*pos < 9999){
		known[*pos][0] = dx;
		known[*pos][1] = dy;
		known[*pos][2] = displacement;
		*pos = *pos+1;
	}
	return displacement;
}

void calcDepth(float *depth, float *left, float *right, int imageWidth, int imageHeight, int featureWidth, int featureHeight, int maximumDisplacement, size_t* floatOps)
{
	memset(depth, 0, imageHeight*imageWidth*sizeof(float));
	float known[9999][3];
	int pos = 0;
	int isOdd = featureWidth % 2;

	for (int y = featureHeight; y < imageHeight - featureHeight; y++)
	{
		for (int x = featureWidth; x < imageWidth - featureWidth; x++)
		{
			int index = y * imageWidth + x;
			// if ((y < featureHeight) || (y >= imageHeight - featureHeight) || (x < featureWidth) || (x >= imageWidth - featureWidth))
			// {
			// 	depth[index] = 0;
			// 	continue;
			// }

			float minimumSquaredDifference = -1;
			int minimumDy = 0;
			int minimumDx = 0;

			for (int dy = -maximumDisplacement; dy <= maximumDisplacement; dy++)
			{
				for (int dx = -maximumDisplacement; dx <= maximumDisplacement; dx++)
				{
					if (y + dy - featureHeight < 0 || y + dy + featureHeight >= imageHeight || x + dx - featureWidth < 0 || x + dx + featureWidth >= imageWidth)
					{
						continue;
					}

					float squaredDifference = 0;
					__m128 temp = _mm_setzero_ps();
					for (int boxY = -featureHeight; boxY <= featureHeight; boxY++)
					{
						/***
						for (int boxX = -featureWidth; boxX <= featureWidth; boxX++)
						{
							int leftX = x + boxX;
							int leftY = y + boxY;
							int rightX = x + dx + boxX;
							int rightY = y + dy + boxY;

							float difference = left[leftY * imageWidth + leftX] - right[rightY * imageWidth + rightX];
							squaredDifference += difference * difference;

							if (floatOps != NULL)
							{
								*floatOps += 3;
							}
						***/
						__m128 result;
						for (int boxX = 0; boxX < 2*featureWidth / 4 * 4; boxX += 4)
						{
							int leftX = x + boxX - featureWidth;
							int leftY = y + boxY;
							int rightX = x + dx + boxX - featureWidth;
							int rightY = y + dy + boxY;

							result = _mm_loadu_ps(&left[leftY * imageWidth + leftX]);
							result = _mm_sub_ps(result, _mm_loadu_ps(&right[rightY * imageWidth + rightX]));
							result = _mm_mul_ps(result, result);
							temp = _mm_add_ps(temp, result);
						}
						/***
						for (int boxX = 2*featureWidth / 4 * 4; boxX <= 2*featureWidth; boxX++)
						{
							int leftX = x + boxX - featureWidth;
							int leftY = y + boxY;
							int rightX = x + dx + boxX - featureWidth;
							int rightY = y + dy + boxY;

							float difference = left[leftY * imageWidth + leftX] - right[rightY * imageWidth + rightX];
							squaredDifference += difference * difference;
						}
						***/
						int leftX = x + 2*featureWidth / 4 * 4 - featureWidth;
                        int leftY = y + boxY;
                        int rightX = x + dx + 2*featureWidth / 4 * 4 - featureWidth;
                       	int rightY = y + dy + boxY;
						if(isOdd)
						{
							int leftX = x + 2*featureWidth / 4 * 4 - featureWidth;
                            int leftY = y + boxY;
                            int rightX = x + dx + 2*featureWidth / 4 * 4 - featureWidth;
                            int rightY = y + dy + boxY;
							result = _mm_loadu_ps(&left[leftY * imageWidth + leftX]);
                            result = _mm_sub_ps(result, _mm_loadu_ps(&right[rightY * imageWidth + rightX]));
                            result = _mm_mul_ps(result, result);
							float array[4];
	                		_mm_storeu_ps((float *)&array, result);
	                		squaredDifference += array[0];
	                		squaredDifference += array[1];
	                		squaredDifference += array[2];
						}
						else
						{
							float difference = left[leftY * imageWidth + leftX] - right[rightY * imageWidth + rightX];
                            squaredDifference += difference * difference;
						}
					}
					float array[4];
					_mm_storeu_ps((float *)&array, temp);
					squaredDifference += array[0];
					squaredDifference += array[1];
					squaredDifference += array[2];
					squaredDifference += array[3];

					if ((minimumSquaredDifference == -1) || ((minimumSquaredDifference == squaredDifference) && (displacementNaive(dx, dy) < displacementNaive(minimumDx, minimumDy))) || (minimumSquaredDifference > squaredDifference))
					{
						minimumSquaredDifference = squaredDifference;
						minimumDx = dx;
						minimumDy = dy;
					}
				}
			}

			if (minimumSquaredDifference != -1)
			{
				if (maximumDisplacement == 0)
				{
					depth[index] = 0;
				}
				else
				{
					depth[y * imageWidth + x] = displacementNaive(minimumDx, minimumDy);
				}
			}
			else
			{
				depth[index] = 0;
			}
		}
	}
}


void calcDepthOptimized(float *depth, float *left, float *right, int imageWidth, int imageHeight, int featureWidth, int featureHeight, int maximumDisplacement)
{
	calcDepth(depth, left, right, imageWidth, imageHeight, featureWidth, featureHeight, maximumDisplacement, NULL);
}
