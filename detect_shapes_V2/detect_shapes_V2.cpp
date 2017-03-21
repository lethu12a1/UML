/**
* Simple shape detector program.
* It loads an image and tries to find simple shapes (rectangle, triangle, circle, etc) in it.
* This program is a modified version of `squares.cpp` found in the OpenCV sample dir.
*/
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <cmath>
#include <iostream>
using namespace cv;
using namespace std;

/**
* Helper function to find a cosine of angle between vectors
* from pt0->pt1 and pt0->pt2
*/
static double angle(cv::Point pt1, cv::Point pt2, cv::Point pt0)
{
	double dx1 = pt1.x - pt0.x;
	double dy1 = pt1.y - pt0.y;
	double dx2 = pt2.x - pt0.x;
	double dy2 = pt2.y - pt0.y;
	return (dx1*dx2 + dy1*dy2) / sqrt((dx1*dx1 + dy1*dy1)*(dx2*dx2 + dy2*dy2) + 1e-10);
}

/**
* Helper function to display text in the center of a contour
*/
void setLabel(cv::Mat& im, const std::string label, std::vector<cv::Point>& contour)
{
	int fontface = cv::FONT_HERSHEY_SIMPLEX;
	double scale = 0.4;
	int thickness = 1;
	int baseline = 0;

	cv::Size text = cv::getTextSize(label, fontface, scale, thickness, &baseline);
	cv::Rect r = cv::boundingRect(contour);

	cv::Point pt(r.x + ((r.width - text.width) / 2), r.y + ((r.height + text.height) / 2));
	if (label == "RECT")
	cv::rectangle(im, pt + cv::Point(0, baseline), pt + cv::Point(text.width, -text.height), CV_RGB(255, 0, 0), CV_FILLED);
	else if (label == "CIR")
		cv::rectangle(im, pt + cv::Point(0, baseline), pt + cv::Point(text.width, -text.height), CV_RGB(0, 255, 0), CV_FILLED);
	else if (label == "TRI")
		cv::rectangle(im, pt + cv::Point(0, baseline), pt + cv::Point(text.width, -text.height), CV_RGB(0, 0, 255), CV_FILLED);
	cv::putText(im, label, pt, fontface, scale, CV_RGB(0, 0, 255), thickness, 8);
}

int main()
{
	//cv::Mat src = cv::imread("polygon.png");
	string dirname = "D:\\truong trinh hoc\\mining UML\\sequencediagram";
	vector<String> filenames;
	glob(dirname, filenames);//doc tat ca cac anh trong folder.
	namedWindow("src", 2);
	namedWindow("dst", 2);
	resizeWindow("src", 600, 600);
	resizeWindow("dst", 600, 600);
	for (size_t t = 0; t < filenames.size(); t++){
		cv::Mat src = cv::imread(filenames[t]);
		if (src.empty())
			return -1;

		// Convert to grayscale
		cv::Mat gray;
		cv::cvtColor(src, gray, CV_BGR2GRAY);
		GaussianBlur(gray, gray, Size(11, 11), 0, 0);
		// Use Canny instead of threshold to catch squares with gradient shading
		cv::Mat bw;
		//cv::Canny(gray, bw, 0, 50, 5);
		//threshold(gray, bw, 200, 255, CV_THRESH_BINARY_INV);
		adaptiveThreshold(gray, bw, 255, 0, 0, 11, -2);
		floodFill(bw, Point(10, 10), Scalar(255));
		floodFill(bw, Point(10, 10), Scalar(0));
		imshow("floodfill", bw);
		// Find contours
		std::vector<std::vector<cv::Point> > contours;
		cv::findContours(bw.clone(), contours, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_SIMPLE);

		std::vector<cv::Point> approx;
		cv::Mat dst = src.clone();

		for (int i = 0; i < contours.size(); i++)
		{
			// Approximate contour with accuracy proportional
			// to the contour perimeter
			cv::approxPolyDP(cv::Mat(contours[i]), approx, cv::arcLength(cv::Mat(contours[i]), true)*0.02, true);

			// Skip small or non-convex objects 
			if (std::fabs(cv::contourArea(contours[i])) < 100 || !cv::isContourConvex(approx))
				continue;

			if (approx.size() == 3)
			{
				setLabel(src, "TRI", contours[i]);    // Triangles
			}
			else if (approx.size() >= 4 && approx.size() <= 6)
			{
				// Number of vertices of polygonal curve
				int vtc = approx.size();

				// Get the cosines of all corners
				std::vector<double> cos;
				for (int j = 2; j < vtc + 1; j++)
					cos.push_back(angle(approx[j%vtc], approx[j - 2], approx[j - 1]));

				// Sort ascending the cosine values
				std::sort(cos.begin(), cos.end());

				// Get the lowest and the highest cosine
				double mincos = cos.front();
				double maxcos = cos.back();

				// Use the degrees obtained above and the number of vertices
				// to determine the shape of the contour
				if (vtc == 4 && mincos >= -0.1 && maxcos <= 0.3)
					setLabel(src, "RECT", contours[i]);
				else if (vtc == 5 && mincos >= -0.34 && maxcos <= -0.27)
					setLabel(dst, "PENTA", contours[i]);
				else if (vtc == 6 && mincos >= -0.55 && maxcos <= -0.45)
					setLabel(dst, "HEXA", contours[i]);
			}
			else
			{
				// Detect and label circles
				double area = cv::contourArea(contours[i]);
				cv::Rect r = cv::boundingRect(contours[i]);
				int radius = r.width / 2;

				if (std::abs(1 - ((double)r.width / r.height)) <= 0.2 &&
					std::abs(1 - (area / (CV_PI * std::pow(radius, 2)))) <= 0.2)
					setLabel(src, "CIR", contours[i]);
			}
		}

		cv::imshow("src", src);
		cv::imshow("dst", bw);
		if (cv::waitKey(0) == 27) break;
	}
	return 0;
}

