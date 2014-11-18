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
#include <float.h>

#define ABS(x) (((x) < 0) ? (-(x)) : (x))

void calcDepthOptimized(float *depth, float *left, float *right, int imageWidth, int imageHeight, int featureWidth, int featureHeight, int maximumDisplacement)
{
	memset(depth, 0, imageHeight*imageWidth*sizeof(float));
	int isOdd = featureWidth % 2;

	for (int y = featureHeight; y < imageHeight - featureHeight; y++)
	{
		for (int x = featureWidth; x < imageWidth - featureWidth; x++)
		{
			int index = y * imageWidth + x;
			float minimumSquaredDifference = FLT_MAX;
			int minimumDy = 0;
			int minimumDx = 0;

			int temp1 = 2*featureWidth-(2*featureWidth)%4;
			for (int dy = -maximumDisplacement; dy <= maximumDisplacement; dy++)
			{
				for (int dx = -maximumDisplacement; dx <= maximumDisplacement; dx++)
				{
					if (y + dy - featureHeight < 0 || y + dy + featureHeight >= imageHeight || x + dx - featureWidth < 0 || x + dx + featureWidth >= imageWidth)
					{
						continue;
					}
					float squaredDifference = 0;
					__m128 result;	
					__m128 temp = _mm_setzero_ps();
					int boxY, boxX;
					for (boxY = -featureHeight; boxY <= featureHeight; boxY++)
					{
						for (boxX = 0; boxX < temp1; boxX += 4)
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
					}
					float array[4];
					_mm_storeu_ps((float *)&array, temp);
					squaredDifference += array[0]+array[1]+array[2]+array[3];
					if(minimumSquaredDifference >= squaredDifference)
					{
						if(isOdd)
						{
							for(boxY = -featureHeight; boxY <= featureHeight; boxY++)
							{
								int leftX = x + featureWidth-2;
								int leftY = y + boxY;
								int rightX = x + dx + featureWidth-2;
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
						}
						else
						{
							for(boxY = -featureHeight; boxY <= featureHeight; boxY++)
							{
								int leftX = x + featureWidth;
								int leftY = y + boxY;
								int rightX = x + dx + featureWidth;
								int rightY = y + dy + boxY;
								float difference = left[leftY * imageWidth + leftX] - right[rightY * imageWidth + rightX];
								squaredDifference += difference * difference;
                        	}
						}
					}
					if ((minimumSquaredDifference > squaredDifference) || ((minimumSquaredDifference == squaredDifference) && (displacementNaive(dx, dy) < displacementNaive(minimumDx, minimumDy))))
					{
						minimumSquaredDifference = squaredDifference;
						minimumDx = dx;
						minimumDy = dy;
					}
				}
			}
			if((minimumSquaredDifference == FLT_MAX) || (maximumDisplacement == 0))
			{
				depth[index] = 0;
			}
			else
			{
				depth[index] = displacementNaive(minimumDx, minimumDy);
			}
		}
	}
}
