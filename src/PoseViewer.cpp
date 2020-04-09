#include "PoseViewer.h"

 //构造函数
PoseViewer::PoseViewer():mbInit(false)
{
    mWinW = 1080; 
    mWinH = 768;  
}
    //初始化
void PoseViewer::init()
{
    if(mbInit)
        return;
    pangolin::CreateWindowAndBind("Simulator",mWinW,mWinH);
    // 3D Mouse handler requires depth testing to be enabled
    glEnable(GL_DEPTH_TEST);
    // Issue specific OpenGl we might need
    glEnable (GL_BLEND);
    glBlendFunc (GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
    pangolin::View &menuPanel = pangolin::CreatePanel("menu").
                                SetBounds(pangolin::Attach::Pix(0),pangolin::Attach::Pix(25),0.0,pangolin::Attach::Pix(mWinW));
    menuPanel.SetLayout(pangolin::LayoutEqualHorizontal);
    
    mViewF = 800;
    mViewX = 0;
    mViewY = -10;
    mViewZ = -0.1;
    mCam = pangolin::OpenGlRenderState(
                pangolin::ProjectionMatrix(mWinW,mWinH,mViewF,mViewF,(mWinW >> 1),(mWinH >> 1),0.1,5000),
                pangolin::ModelViewLookAt(mViewX,mViewY,mViewZ, 0,0,0,0.0,-1.0, 0.0)
                );
    // Add named OpenGL viewport to window and provide 3D Handler
    mpView = &pangolin::CreateDisplay()
            .SetBounds(pangolin::Attach::Pix(25), 1.0, 0, 1.0, -(float)mWinW/mWinH)
            .SetHandler(new pangolin::Handler3D(mCam));
    
    mbInit = true;
}
void PoseViewer::renderLoop()
{
    init();
    while(renderOnce())
    {
        ;
    }
}
    //绘制
bool PoseViewer::renderOnce()
{
    if(!mbInit)
        return false;
    static pangolin::Var<bool> menuFollowCamera("menu.Follow Camera",true,true);
    static pangolin::Var<bool> menuShowPoints("menu.MapPoints",true,true);
    static pangolin::Var<bool> menuShowKeyFrames("menu.MapFrames",true,true);
    static pangolin::Var<bool> menuShowGraph("menu.CovGraph",true,true);
    static pangolin::Var<bool> menuShowLines("menu.RelLines",true,true);
    pangolin::OpenGlMatrix Twc;
    Twc.SetIdentity();
    static bool bFollow = true;
    {
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
       
        const float clr = 0.1f;
        glClearColor(clr, clr, clr ,1.0f);
        
        if(menuFollowCamera && bFollow)
        {
            mCam.Follow(Twc);
        }
        else if(menuFollowCamera && !bFollow) 
        {
            mCam.SetModelViewMatrix(pangolin::ModelViewLookAt(mViewX, mViewY, mViewZ, 0, 0, 0, 0.0, -1.0, 0.0));
            mCam.Follow(Twc);
            bFollow = true;
        }
        else
        {
            bFollow = false;
        }
        mpView->Activate(mCam);
        if(menuShowKeyFrames)
        {
            drawFrames();
        }
        if(menuShowPoints)
        {
            drawMapPoints();
        }
        pangolin::FinishFrame();
    }
    return !pangolin::ShouldQuit();
}
void Nor(Mat &pt)
{
    float d = cv::norm(pt);
    pt = pt / d;
}
//绘制坐标轴
static void drawCoordinateAxis(const cv::Point3f &pt)
{
    glLineWidth(2.0);
   
    glBegin(GL_LINES);
    glColor3f(1.0f,0.0f,0.0f);
    glVertex3f(pt.x,pt.y,pt.z);
    glVertex3f(1,0,0);
    glColor3f(0.0f,1.0f,0.0f);
    glVertex3f(pt.x,pt.y,pt.z);
    glVertex3f(0,1,0);
    glColor3f(0.0f,0.0f,1.0f);
    glVertex3f(pt.x,pt.y,pt.z);
    glVertex3f(0,0,1);
    glEnd();
}

inline cv::Mat GetWdPt(const cv::Mat &pose)
{
    Mat twl = pose.clone();
    cv::Mat Rcw = twl.rowRange(0,3).colRange(0,3);
    cv::Mat tcw = twl.rowRange(0,3).col(3);
    cv::Mat Rwc = Rcw.t();
    cv::Mat Ow = -Rwc*tcw; 
    return Ow;
}

static void drawPose(const cv::Mat &pose)
{
    cv::Mat Ow = GetWdPt(pose);
    cv::Mat Rwc = pose.rowRange(0,3).colRange(0,3).t();
    cv::Mat mWorldPosInv = cv::Mat::eye(4,4,CV_64F);
    Rwc.copyTo(mWorldPosInv.rowRange(0,3).colRange(0,3));
    Ow.copyTo(mWorldPosInv.rowRange(0,3).col(3));
    Mat t = mWorldPosInv.t();
    const float w = 0.5;
    const float h = w * 0.75;
    const float z = w * 0.6;
    glPushMatrix();
    glMultMatrixd(t.ptr<GLdouble>(0));
    glLineWidth(1);
    glColor3f(0.0f,1.0f,1.0f);
    glBegin(GL_LINES);
    glVertex3f(0,0,0);
    glVertex3f(w,h,z);
    glVertex3f(0,0,0);
    glVertex3f(w,-h,z);
    glVertex3f(0,0,0);
    glVertex3f(-w,-h,z);
    glVertex3f(0,0,0);
    glVertex3f(-w,h,z);
    glVertex3f(w,h,z);
    glVertex3f(w,-h,z);
    glVertex3f(-w,h,z);
    glVertex3f(-w,-h,z);
    glVertex3f(-w,h,z);
    glVertex3f(w,h,z);
    glVertex3f(-w,-h,z);
    glVertex3f(w,-h,z);
    glEnd();
    glPopMatrix();
}
//绘制帧
void PoseViewer::drawFrames()
{
    drawCoordinateAxis(Point3f(0,0,0));
    const float w = 0.5;
    const float h = w * 0.75;
    const float z = w * 0.6;
    const float mKeyFrameLineWidth = 1.0;
 
    if(mFrames.empty())
        return;

    for(size_t i = 0; i < mFrames.size(); ++i)
    {
        drawPose(mFrames[i]._pose);
    }

    glLineWidth(mKeyFrameLineWidth);
    glBegin(GL_LINE_STRIP);
    glColor3f(0.0f,1.0f,0.0f);
    glVertex3f(0,0,0);
    for(size_t i = 1;i < mFrames.size(); ++i)
    {
        Mat p = GetWdPt(mFrames[i]._pose);
        glVertex3f((float)p.at<double>(0),
                   (float)p.at<double>(1),
                   (float)p.at<double>(2));
    }
    glEnd();
}
//绘制地图
void PoseViewer::drawMapPoints()
{
    if(mPoints.empty())
        return ;

    glPointSize(2.0);
    glBegin(GL_POINTS);
    glColor3f(0.6,0.8,0.0);
    for(size_t i = 0;i < mPoints.size();++i)
    {
        cv::Mat pos = mPoints[i]._pose;
        glVertex3f(pos.at<double>(0),
                   pos.at<double>(1),
                   pos.at<double>(2));
    }
    glEnd();
}