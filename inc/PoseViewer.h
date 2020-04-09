#include <pangolin/pangolin.h>
#include <opencv2/opencv.hpp>
#include <vector>
using namespace std;
using namespace cv;
 //pangolin可视化
class PoseViewer 
{
public:
    //Node
    struct  PoseNode 
    {
        Mat _pose;
    };

    typedef std::vector<PoseNode> NodeVector;
    typedef NodeVector::iterator  NodeVIter;

public:
    //构造函数
    PoseViewer();

    //加入节点
    void push_node(const PoseNode &node)
    {
        if(!node._pose.empty())
        {
            if(node._pose.rows > 2)
            {
                mFrames.emplace_back(node);
            }
            else
            {
                mPoints.emplace_back(node);
            }
            
        }
    }


     //初始化
    virtual void init();
    //绘制一次
    virtual bool renderOnce();
    //绘制循环
    virtual void renderLoop();
protected:
    //绘制帧
    void drawFrames();
    //绘制地图点
    void drawMapPoints();
protected:
    pangolin::View                 *mpView;
    pangolin::OpenGlRenderState     mCam;
    float                           mViewF;
    float                           mViewX;
    float                           mViewY;
    float                           mViewZ;
    bool                            mbInit;
    int                             mWinW;
    int                             mWinH;
    
    NodeVector                      mFrames;
    NodeVector                      mPoints;
};