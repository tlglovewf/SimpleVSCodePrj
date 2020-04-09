#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/xfeatures2d.hpp>

#include <map>
#include <vector>

#include <fstream>

using namespace std;
using namespace cv;
using namespace cv::xfeatures2d;


struct BatchValue
{
    std::string _btname;
    int _n;
    std::vector<string> _names;
    std::vector<Mat>    _poses;
    bool isvaild()const
    {
        return _names.size() == _poses.size();
    }
    BatchValue(const std::string &btn,int n):_btname(btn),_n(n)
    {
        _names.reserve(n);
        _poses.reserve(n);
    }
};
typedef vector<BatchValue>              SlamBatchMap;
typedef SlamBatchMap::iterator          SlamBMIter;

#define BEGINFILEREGION(PATH,MD)  try                               \
                                  {                                 \
                                     if(open(PATH,std::ios::MD))    \
                                     {

#define ENDFILEREGION()               mfile.close();                \
                                     }                               \
                                  }                                 \
                                  catch(const std::exception& e)    \
                                  {                                 \
                                      std::cerr << e.what() << '\n';\
                                  }



class HdBatch
{
public:
    void loadBatchFiles(const std::string &path)
    {
        assert(!path.empty());
        BEGINFILEREGION(path,in)
        
        while(!mfile.eof())
        {
            std::string line;
            getline(mfile,line);
            char batchname[20] = {0};
            int  n;
            sscanf(line.c_str(),"%s %d",batchname,&n);
            if(n < 1)
                continue;
            int index = 0;

            
            BatchValue pv(batchname,n);
            //read batch files name
            for(int i = 0;i < n; ++i)
            {
                getline(mfile,line);
                pv._names.emplace_back(line);
                Mat pose = Mat::eye(4,4,CV_64F);
                pose.at<double>(2,3) = i;
                pv._poses.emplace_back(pose);
            }
            mBatches.emplace_back(pv);
        }
        ENDFILEREGION()
    }
    void writeBatchResult(const std::string &outpath)
    {   
        assert(!outpath.empty());
        BEGINFILEREGION(outpath,out)
        
        //head    
        mfile << std::setiosflags(std::ios::fixed) << std::setiosflags(std::ios::right)
              << std::setw(9)  << "ImageName"
              << std::setw(7)  << "valid"
              << std::setw(15) << "R11"
              << std::setw(15) << "R12"
              << std::setw(15) << "R13"
              << std::setw(15) << "R21"
              << std::setw(15) << "R22"
              << std::setw(15) << "R23"
              << std::setw(15) << "R31"
              << std::setw(15) << "R32"
              << std::setw(15) << "R33"
              << std::setw(15) << "T1"
              << std::setw(15) << "T2"
              << std::setw(15) << "T3"
              << std::endl;
        
        SlamBMIter it = mBatches.begin();
        SlamBMIter ed = mBatches.end();
        for(;it != ed; ++it)
        {
            if(it->isvaild())
            {
                //batch info
                mfile << it->_btname.c_str() << " " << it->_n << std::endl;
                for(int i = 0;i < it->_n; ++i)
                {
                    //pose
                    mfile << std::setiosflags(std::ios::fixed) << std::setiosflags(std::ios::right)
                          << std::setw(9)  << it->_names[i].c_str()
                          << std::setw(7)  << (!it->_poses.empty());

                    const Mat &pose =  it->_poses[i];
                    double scale = 1.0;
                    mfile << std::setiosflags(std::ios::fixed) << std::setprecision(9) << std::setiosflags(std::ios::right)
                          << std::setw(15) << pose.at<double>(0,0)
                          << std::setw(15) << pose.at<double>(0,1)
                          << std::setw(15) << pose.at<double>(0,2)
                          << std::setw(15) << pose.at<double>(1,0)
                          << std::setw(15) << pose.at<double>(1,1)
                          << std::setw(15) << pose.at<double>(1,2)
                          << std::setw(15) << pose.at<double>(2,0)
                          << std::setw(15) << pose.at<double>(2,1)
                          << std::setw(15) << pose.at<double>(2,2)
                          << std::setw(15) << pose.at<double>(0,3) * scale
                          << std::setw(15) << pose.at<double>(1,3) * scale
                          << std::setw(15) << pose.at<double>(2,3) * scale
                          << std::endl;
                }
            }
        }


        ENDFILEREGION()
    }
protected:
     //打开文件
     bool open(const std::string &path,std::ios::openmode type)
     {
         mfile.open(path, type);
         return mfile.is_open();
     }

protected:
    fstream mfile;
    SlamBatchMap mBatches;
};

typedef vector<KeyPoint>        KeyPtVector;
typedef KeyPtVector::iterator   KeyPtVIter;

void detect(const Ptr<Feature2D> &feature,const std::string &path, KeyPtVector &keypts,Mat &out)
{
    out = imread(path);
    assert(!out.empty());
    feature->detect(out,keypts);
}

Mat drawKeys(const Mat &img, const KeyPtVector &keys)
{
    Mat keypoint_img;
    drawKeypoints(img,keys,keypoint_img,CV_RGB(0,0,255),DrawMatchesFlags::DEFAULT);
    putText(keypoint_img,"key size:" + std::to_string(keys.size()),cv::Point2f(50,50),CV_FONT_HERSHEY_COMPLEX, 1, CV_RGB(255,0,0), 2, CV_AA);
    return keypoint_img;
}


//及时
class Time_Interval
{
public:
    /*  开始
     */
    inline void start()
    {
        _t = clock();
    }
    /*  结束
     */
    inline float end()
    {
        return (clock() - _t) / (float)CLOCKS_PER_SEC;
    }
    /* 输出
	 */
    inline void prompt(const std::string &str,bool isreset = false)
    {
        std::cout << str.c_str() << end() << "s" << std::endl;
        if(isreset)
            start();
    }

protected:
    time_t _t;
};

#define FEATUREMATCH(X)         void feature##X##Match(const Mat &des1, const Mat &des2, vector<DMatch> &matches,vector<DMatch> &good_matches)
#define FEATUREMATCHFUNC(X)     case e##X##Type:\
                                return feature##X##Match(des1,des2,matches,good_matches);
#define FEATUREMATCHTYPE(X)  e##X##Type

enum eFeatureMType
{
    FEATUREMATCHTYPE(BF),
    FEATUREMATCHTYPE(Flann),
    FEATUREMATCHTYPE(Knn)
};
FEATUREMATCH(BF)
{
    Ptr<DescriptorMatcher> matcher = DescriptorMatcher::create(DescriptorMatcher::BRUTEFORCE);  
    matcher->match(des1,des2,matches);

    double min_dist = 10000,max_dist = 0;

    for(int i = 0; i < des1.rows; ++i)
    {
        double dist = matches[i].distance;
        if( dist < min_dist ) min_dist = dist;
        if( dist > max_dist ) max_dist = dist;
    }

    min_dist = min_element(matches.begin(), matches.end(),[](const DMatch &m1, const DMatch &m2){return m1.distance < m2.distance;})->distance;
    max_dist = max_element(matches.begin(), matches.end(),[](const DMatch &m1, const DMatch &m2){return m1.distance < m2.distance;})->distance;

    printf("-- min dist : %f \n",min_dist);
    printf("-- max dist : %f \n",max_dist);

    for(int i = 0; i < des1.rows; ++i)
    {
        if(matches[i].distance < max( 2 * min_dist, 30.0))
        {
            good_matches.emplace_back(matches[i]);
        }
    }
}

FEATUREMATCH(Flann)
{
    cv::Ptr<cv::DescriptorMatcher> matcher = cv::DescriptorMatcher::create("FlannBased");
    matcher->match(des1,des2,matches);
     double maxDist = 10;
    for(int i =0;i<des1.rows;i++)
    {
        double dist = matches[i].distance;
        if(dist>maxDist)
            maxDist= dist;
    }

    for(int i =0;i<des1.rows;i++)
    {
        if(matches[i].distance<0.5*maxDist)             ////调参褚   0.1越小 越精确  官方推荐0.5 如果确定点 可改变
        {
            good_matches.push_back(matches[i]);
        }
    }
}

FEATUREMATCH(Knn)
{
    // cv::Ptr<cv::DescriptorMatcher> matcher = cv::DescriptorMatcher::create("FlannBased");
    cv::Ptr<cv::DescriptorMatcher> matcher = cv::DescriptorMatcher::create(DescriptorMatcher::BRUTEFORCE_SL2);
    vector< vector<DMatch> > knnMatches;
    matcher->knnMatch(des1,des2,knnMatches,2);
    // matcher->radiusMatch()
    const float minRatio = 0.4f;//0.4  0.5  0.6

    for(int i = 0;i < knnMatches.size();++i)
    {
        const DMatch &bestmatch   = knnMatches[i][0];
        const DMatch &bettermatch = knnMatches[i][1];

        float distanceRatio = bestmatch.distance / bettermatch.distance;
        if(distanceRatio < minRatio)
        {
            good_matches.push_back(bestmatch);
        }
    }
}


void featureMatch(eFeatureMType etype, const Mat &des1, const Mat &des2, vector<DMatch> &matches, vector<DMatch> &good_matches)
{
    switch (etype)
    {
        FEATUREMATCHFUNC(BF)
        FEATUREMATCHFUNC(Flann)
        FEATUREMATCHFUNC(Knn)   
    }
}






int main(void)
{
    //（1）nfeatures，保留的最佳特性的数量。特征按其得分进行排序(以SIFT算法作为局部对比度进行测量)；
    //（2）nOctavelLayers，高斯金字塔最小层级数，由图像自动计算出；
    //（3）constrastThreshold，对比度阈值用于过滤区域中的弱特征。阈值越大，检测器产生的特征越少。；
    //（4）edgeThreshold ，用于过滤掉类似边缘特征的阈值。 请注意，其含义与contrastThreshold不同，即edgeThreshold越大，滤出的特征越少；
    //（5）sigma，高斯输入层级， 如果图像分辨率较低，则可能需要减少数值。
    Ptr<SIFT> featurer = SIFT::create(40,4,0.01,10,1.4);

    Mat img1;
    Mat img2;
    KeyPtVector keypt1;
    KeyPtVector keypt2;
    Mat des1;
    Mat des2;
    Time_Interval time;
    time.start();
    detect(featurer,"/media/tlg/work/tlgfiles/HDData/0326-2/Image-0/0_13244.jpg",keypt1,img1);
    time.prompt("feature detect:");
    
    detect(featurer,"/media/tlg/work/tlgfiles/HDData/0326-2/Image-0/0_13245.jpg",keypt2,img2);
    
    Mat keypoint_img = drawKeys(img1,keypt1);
    imwrite("/media/tlg/work/tlgfiles/HDData/result/sift_13244.jpg",keypoint_img);
    keypoint_img = drawKeys(img2,keypt2);
    imwrite("/media/tlg/work/tlgfiles/HDData/result/sift_13245.jpg",keypoint_img);
    time.start();
    featurer->compute(img1,keypt1,des1);
    time.prompt("calc descirpt ");
    featurer->compute(img2,keypt2,des2);

    vector<DMatch> matches;
   
    std::vector<DMatch> good_matches;
    
    time.start();
    // featureMatch(eBFType,des1,des2,matches,good_matches);
    // featureMatch(eFlannType,des1,des2,matches,good_matches);
    featureMatch(eKnnType,des1,des2,matches,good_matches);
    time.prompt("feature matching cost ");


    vector<Point2f> pt1s;
    vector<Point2f> pt2s;


    for(int i = 0; i < good_matches.size(); ++i)
    {
        pt1s.push_back( keypt1[good_matches[i].queryIdx].pt);
        pt2s.push_back( keypt2[good_matches[i].trainIdx].pt);
       
    }

    Mat stats;
    Mat F = cv::findFundamentalMat(pt1s,pt2s,stats,FM_RANSAC);


    cout << "-----" << endl << F << endl;
    F = cv::findFundamentalMat(pt1s,pt2s,stats,FM_8POINT);
    cout << "-----" << endl  << F << endl; 

    //save images
    Mat img_match;
    Mat img_goodmatch;
    drawMatches(img1, keypt1, img2, keypt2,matches ,img_match,CV_RGB(255,0,0),CV_RGB(0,0,255));
    putText(img_match,"match size:" + std::to_string(matches.size()),cv::Point2f(50,50),CV_FONT_HERSHEY_COMPLEX, 1, CV_RGB(255,0,0), 2, CV_AA);
    drawMatches(img1, keypt1, img2, keypt2,good_matches ,img_goodmatch,CV_RGB(255,0,0),CV_RGB(0,0,255));
    putText(img_goodmatch,"match size:" + std::to_string(good_matches.size()),cv::Point2f(50,50),CV_FONT_HERSHEY_COMPLEX, 1, CV_RGB(255,0,0), 2, CV_AA);


    // imwrite("/media/tlg/work/tlgfiles/HDData/result/bf_match.jpg",img_match);
    // imwrite("/media/tlg/work/tlgfiles/HDData/result/bf_good_match.jpg",img_goodmatch);

    // imwrite("/media/tlg/work/tlgfiles/HDData/result/flann_match.jpg",img_match);
    // imwrite("/media/tlg/work/tlgfiles/HDData/result/flann_good_match.jpg",img_goodmatch);

    imwrite("/media/tlg/work/tlgfiles/HDData/result/knn_match.jpg",img_match);
    imwrite("/media/tlg/work/tlgfiles/HDData/result/knn_good_match.jpg",img_goodmatch);

    



    cout << "write successfully.." << endl;
    return 0;
}