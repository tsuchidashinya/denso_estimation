#include <accuracy_package/accuracy_server.hpp>

int main(int argc, char** argv)
{
    ros::init(argc, argv, "accuracy_server");
    ros::NodeHandle nh;
    AccuracyServer acc_server(nh);
    ros::spin();
    return 0;
}