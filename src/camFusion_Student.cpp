
#include <iostream>
#include <algorithm>
#include <numeric>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include "camFusion.hpp"
#include "dataStructures.h"

using namespace std;


// Create groups of Lidar points whose projection into the camera falls into the same bounding box
void clusterLidarWithROI(std::vector<BoundingBox> &boundingBoxes, std::vector<LidarPoint> &lidarPoints, float shrinkFactor, cv::Mat &P_rect_xx, cv::Mat &R_rect_xx, cv::Mat &RT)
{
    // loop over all Lidar points and associate them to a 2D bounding box
    cv::Mat X(4, 1, cv::DataType<double>::type);
    cv::Mat Y(3, 1, cv::DataType<double>::type);

    for (auto it1 = lidarPoints.begin(); it1 != lidarPoints.end(); ++it1)
    {
        // assemble vector for matrix-vector-multiplication
        X.at<double>(0, 0) = it1->x;
        X.at<double>(1, 0) = it1->y;
        X.at<double>(2, 0) = it1->z;
        X.at<double>(3, 0) = 1;

        // project Lidar point into camera
        Y = P_rect_xx * R_rect_xx * RT * X;
        cv::Point pt;
        pt.x = Y.at<double>(0, 0) / Y.at<double>(0, 2); // pixel coordinates
        pt.y = Y.at<double>(1, 0) / Y.at<double>(0, 2);

        vector<vector<BoundingBox>::iterator> enclosingBoxes; // pointers to all bounding boxes which enclose the current Lidar point
        for (vector<BoundingBox>::iterator it2 = boundingBoxes.begin(); it2 != boundingBoxes.end(); ++it2)
        {
            // shrink current bounding box slightly to avoid having too many outlier points around the edges
            cv::Rect smallerBox;
            smallerBox.x = (*it2).roi.x + shrinkFactor * (*it2).roi.width / 2.0;
            smallerBox.y = (*it2).roi.y + shrinkFactor * (*it2).roi.height / 2.0;
            smallerBox.width = (*it2).roi.width * (1 - shrinkFactor);
            smallerBox.height = (*it2).roi.height * (1 - shrinkFactor);

            // check wether point is within current bounding box
            if (smallerBox.contains(pt))
            {
                enclosingBoxes.push_back(it2);
            }

        } // eof loop over all bounding boxes

        // check wether point has been enclosed by one or by multiple boxes
        if (enclosingBoxes.size() == 1)
        { 
            // add Lidar point to bounding box
            enclosingBoxes[0]->lidarPoints.push_back(*it1);
        }

    } // eof loop over all Lidar points
}


void show3DObjects(std::vector<BoundingBox> &boundingBoxes, cv::Size worldSize, cv::Size imageSize, bool bWait)
{
    // create topview image
    cv::Mat topviewImg(imageSize, CV_8UC3, cv::Scalar(255, 255, 255));

    for(auto it1=boundingBoxes.begin(); it1!=boundingBoxes.end(); ++it1)
    {
        // create randomized color for current 3D object
        cv::RNG rng(it1->boxID);
        cv::Scalar currColor = cv::Scalar(rng.uniform(0,150), rng.uniform(0, 150), rng.uniform(0, 150));

        // plot Lidar points into top view image
        int top=1e8, left=1e8, bottom=0.0, right=0.0; 
        float xwmin=1e8, ywmin=1e8, ywmax=-1e8;
        for (auto it2 = it1->lidarPoints.begin(); it2 != it1->lidarPoints.end(); ++it2)
        {
            // world coordinates
            float xw = (*it2).x; // world position in m with x facing forward from sensor
            float yw = (*it2).y; // world position in m with y facing left from sensor
            xwmin = xwmin<xw ? xwmin : xw;
            ywmin = ywmin<yw ? ywmin : yw;
            ywmax = ywmax>yw ? ywmax : yw;

            // top-view coordinates
            int y = (-xw * imageSize.height / worldSize.height) + imageSize.height;
            int x = (-yw * imageSize.width / worldSize.width) + imageSize.width / 2;

            // find enclosing rectangle
            top = top<y ? top : y;
            left = left<x ? left : x;
            bottom = bottom>y ? bottom : y;
            right = right>x ? right : x;

            // draw individual point
            cv::circle(topviewImg, cv::Point(x, y), 4, currColor, -1);
        }

        // draw enclosing rectangle
        cv::rectangle(topviewImg, cv::Point(left, top), cv::Point(right, bottom),cv::Scalar(0,0,0), 2);

        // augment object with some key data
        char str1[200], str2[200];
        sprintf(str1, "id=%d, #pts=%d", it1->boxID, (int)it1->lidarPoints.size());
        putText(topviewImg, str1, cv::Point2f(left-250, bottom+50), cv::FONT_ITALIC, 2, currColor);
        sprintf(str2, "xmin=%2.2f m, yw=%2.2f m", xwmin, ywmax-ywmin);
        putText(topviewImg, str2, cv::Point2f(left-250, bottom+125), cv::FONT_ITALIC, 2, currColor);  
    }

    // plot distance markers
    float lineSpacing = 2.0; // gap between distance markers
    int nMarkers = floor(worldSize.height / lineSpacing);
    for (size_t i = 0; i < nMarkers; ++i)
    {
        int y = (-(i * lineSpacing) * imageSize.height / worldSize.height) + imageSize.height;
        cv::line(topviewImg, cv::Point(0, y), cv::Point(imageSize.width, y), cv::Scalar(255, 0, 0));
    }

    // display image
    string windowName = "3D Objects";
    cv::namedWindow(windowName, 1);
    cv::imshow(windowName, topviewImg);

    if(bWait)
    {
        cv::waitKey(0); // wait for key to be pressed
    }
}


// associate a given bounding box with the keypoints it contains
void clusterKptMatchesWithROI(BoundingBox &boundingBox, std::vector<cv::KeyPoint> &kptsPrev, std::vector<cv::KeyPoint> &kptsCurr, std::vector<cv::DMatch> &kptMatches)
{
    // ...
}


// Compute time-to-collision (TTC) based on keypoint correspondences in successive images
void computeTTCCamera(std::vector<cv::KeyPoint> &kptsPrev, std::vector<cv::KeyPoint> &kptsCurr, 
                      std::vector<cv::DMatch> kptMatches, double frameRate, double &TTC, cv::Mat *visImg)
{
    // ...
}


void computeTTCLidar(std::vector<LidarPoint> &lidarPointsPrev,
                     std::vector<LidarPoint> &lidarPointsCurr, double frameRate, double &TTC)
{
    // ...
}


void matchBoundingBoxes(std::vector<cv::DMatch> &matches, std::map<int, int> &bbBestMatches, DataFrame &prevFrame, DataFrame &currFrame)
{
  std::multimap<int, int> mm;
    // iterate through matches to find the corresponding keypoints in prev and current image
    // add the corresponding bounding boxes if found to mm
    // prevFrame keypoints correspond to queryIdx in matches
    // currFrame keypoints correspond to trainIdx in matches
    for (auto match : matches){
        cv::KeyPoint prev_kp = prevFrame.keypoints.at(match.queryIdx);
        cv::KeyPoint curr_kp = currFrame.keypoints.at(match.trainIdx);
        
        std::vector<int> prev_boxid;
        std::vector<int> curr_boxid;
        
        for (auto bbox : prevFrame.boundingBoxes){
            // std::cout << bbox.keypoints.size() << std::endl;
            if(bbox.roi.contains(prev_kp.pt)){
                prev_boxid.push_back(bbox.boxID);  // add box id to list of boxes the keypoint may be included in.
            }
        }

        for (auto bbox : currFrame.boundingBoxes){
            // std::cout << bbox.keypoints.size() << std::endl;
            if(bbox.roi.contains(curr_kp.pt)){
                curr_boxid.push_back(bbox.boxID);  // add box id to list of boxes the keypoint may be included in.
            }
        }

        // now we have list of boxes in prev and curr data frame that enclose the current keypoint match
        //ensure that there is at least 1 bounding box in each of the data frames
        if(prev_boxid.empty() || curr_boxid.empty()){
            std::cout << "no bounding box match found!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!" << std::endl;
            continue;  // no bounding box match is found for the keypoint
        }

        // if code gets here, there is at least 1 bounding box match found. store the possible matches in mm
        for (auto i : curr_boxid){
            for (auto j : prev_boxid){
                mm.insert(std::pair<int, int>(i, j));
            } 
        }
        // std::cout << "____________________________________________________" << std:::endl;
    }

    // iterate through the bounding boxes in the current frame and select the best match for each BB 
    for (auto bbox : currFrame.boundingBoxes){
        int id = bbox.boxID;
        std::pair <std::multimap<int,int>::iterator, std::multimap<int,int>::iterator> ret;
        ret = mm.equal_range(id);

        std::map<int, int> local;
        for (std::multimap<int,int>::iterator it = ret.first; it!= ret.second; ++it){
            // record the <prev frame bb id> and <count how many times it has been linked> to the curr bb
            int prev_frame_bb_id = it->second;
            std::map<int,int>::iterator local_it = local.find(prev_frame_bb_id);
            if (local_it == local.end()){
                local.insert(std::pair<int, int>(prev_frame_bb_id, 1));
            }
            else
            {
                local_it->second = local_it->second + 1;
            }
        }

        //iterate through map to find the prev_bbox_id with max counts, select this as best match
        int max_count = 0;
        int prev_matched_bb_id = -1; 
        for(auto& map_ele : local){
            if (map_ele.second > max_count){
                max_count = map_ele.second;
                prev_matched_bb_id = map_ele.first;
            }
        }

        if(prev_matched_bb_id == -1){
            std::cout << "error in matchingbounding boxes!!!!!!!!!!!!!!!!!!!!!!!!!!!!!" << std::endl;
            continue;
        }    
        bbBestMatches.insert(std::pair<int, int>(id, prev_matched_bb_id));
        std::cout << "mm size = " << bbBestMatches.size() << std::endl;
    }
    // ...
}

