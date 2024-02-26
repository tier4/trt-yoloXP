#include <memory>
#include <string>
#include "yolo_config_parser.h"

#include <filesystem>
#include <iostream>
#include <sys/stat.h>
#include <iostream>
#include <boost/filesystem.hpp>

#include <random>
#include <algorithm>
#include <memory>
#include <string>
#include <utility>
#include <vector>

template <typename ... Args>
std::string format(const std::string& fmt, Args ... args )
{
  size_t len = std::snprintf( nullptr, 0, fmt.c_str(), args ... );
  std::vector<char> buf(len + 1);
  std::snprintf(&buf[0], len + 1, fmt.c_str(), args ... );
  return std::string(&buf[0], &buf[0] + len);
}

template<class T> bool contain(const std::string& s, const T& v) {
  return s.find(v) != std::string::npos;
}

std::string replaceOtherStr(std::string &replacedStr, std::string from, std::string to) {
  const unsigned int pos = replacedStr.find(from);
  const int len = from.length();

  if (pos == std::string::npos || from.empty()) {
    return replacedStr;
  }

  return replacedStr.replace(pos, len, to);
}

std::random_device rd;
std::mt19937 gen(rd());
int random(int low, int high)
{
  std::uniform_int_distribution<> dist(low, high);
  return dist(gen);
}


void save_image(cv::Mat &img, const std::string &dir, const std::string &name)
{
  fs::path p = dir;
  p.append(name);
  std::string dst = p.string();
  std::cout << "##Save " << dst << std::endl;
  cv::imwrite(dst, img);
}

void
write_prediction(std::string dumpPath, std::string filename, std::vector<std::string> names, tensorrt_yolox::ObjectArray objects, int width, int height)
{
  int pos = filename.find_last_of(".");
  std::string body = filename.substr(0, pos);
  std::string dstName = body + ".txt";
  std::ofstream writing_file;
  fs::path p = dumpPath;
  fs::create_directory(p);  
  p.append(dstName);
  writing_file.open(p.string(), std::ios::out);  
  for (const auto & object : objects) {
    const auto left = object.x_offset;
    const auto top = object.y_offset;
    const auto right = std::clamp(left + object.width, 0, width);
    const auto bottom = std::clamp(top + object.height, 0, height);
    std::string writing_text = format("%s %f %d %d %d %d", names[object.type].c_str(), object.score, left, top, right, bottom);
    writing_file << writing_text << std::endl;
  }
  writing_file.close();
}

void
write_label(std::string outputPath, std::string filename, std::vector<std::string> names, tensorrt_yolox::ObjectArray objects, int width, int height)
{
  int pos = filename.find_last_of(".");
  std::string body = filename.substr(0, pos);
  std::string dstName = body + ".txt";
  std::ofstream writing_file;
  fs::path p = outputPath;
  fs::create_directory(p);  
  p.append(dstName);
  writing_file.open(p.string(), std::ios::out);
  std::cout << "Write" << p.string() << std::endl;
  for (const auto & object : objects) {
    const auto left = object.x_offset;
    const auto top = object.y_offset;
    const auto right = std::clamp(left + object.width, 0, width);
    const auto bottom = std::clamp(top + object.height, 0, height);    
    double x = (left + right) / 2.0 / (double)width;
    double y = (top + bottom) / 2.0 / (double)height;
    double w = (right - left) / (double)width;
    double h = (bottom - top) / (double)height;    
    std::string writing_text = format("%d %f %f %f %f", object.type, x, y, w, h);
    writing_file << writing_text << std::endl;
  }
  writing_file.close();
}

int
main(int argc, char* argv[])
{
  gflags::ParseCommandLineFlags(&argc, &argv, true);
  std::string model_path = get_onnx_path();
  std::string precision = get_precision();
  std::string directory = get_directory_path();
  std::string video_path = get_video_path();
  int cam_id = get_camera_id();    
  std::string output_image_path = "dst.png";
  std::string calibration_images = get_calibration_images();
  const int batch = get_batch_size();
  const bool dont_show = is_dont_show();
  const int dla = get_dla_id();
  const bool cuda = get_cuda_flg();
  const double scale = get_scale();
  const bool first = get_fisrt_flg();
  const bool last = get_last_flg();
  const std::string dumpPath = get_dump_path();
  const std::string outputPath = get_output_path();  
  std::vector<std::vector<int>> colormap = get_colormap();
  std::vector<tensorrt_yolox::Colormap> seg_cmap = get_seg_colormap();
  std::string calibType = get_calib_type();//nvinfer1::CalibrationAlgoType::kMINMAX_CALIBRATION;
  bool prof = get_prof_flg();
  double clip = get_clip_value();
  Window_info window_info = get_window_info();
  Cropping_info cropping_info = get_cropping_info();
  cv::Rect roi(cv::Point(cropping_info.x, cropping_info.y), cv::Size(cropping_info.w, cropping_info.h));
      
  const int num_class = get_classes();
  const float thresh = (float)get_score_thresh();
  const float nms_thresh = 0.7;
  const tensorrt_common::BatchConfig & batch_config = {1, batch/2, batch};
  const size_t workspace_size = (1 << 30);
  tensorrt_common::BuildConfig build_config(
					    calibType, dla, first, last,
					    prof, clip);
  auto trt_yolox = std::make_unique<tensorrt_yolox::TrtYoloX>(model_path, precision, num_class, thresh, nms_thresh, build_config, cuda, calibration_images, scale, "", batch_config, workspace_size);

  bool init = true;
  bool flg_save = getSaveDetections();
  std::string save_path = getSaveDetectionsPath();

  if (flg_save) {
    fs::create_directory(save_path);
    fs::path p = save_path;
    p.append("detections");
    fs::create_directory(p);

    fs::create_directory(save_path);
    fs::path ps = save_path;
    ps.append("segmentations");
    fs::create_directory(ps);    
  }
  
  if (directory != "") {
    std::vector<std::string> filenames;
    for (const auto & file : std::filesystem::directory_iterator(directory)) {
      filenames.push_back(file.path());
    }    
    int iter = getMultiScaleInference() ? 1 : batch;    
    for (int i = 0; i < (int)filenames.size(); i = i + iter) {
      std::vector<cv::Mat> inputs;
      std::vector<cv::Rect> rois;
      std::cout << "Batching ..." << std::endl;
      for (int b = 0; b < iter; b++) {
	cv::Mat image;
	if ((i+b) > ((int)filenames.size()-1)) {
	  std::cout << "Dummy Image" << std::endl;
	  cv::Mat dummy(cv::Size(224, 224), CV_8UC3);
	  inputs.push_back(dummy);
	} else {
	  std::cout << filenames[i+b] << std::endl;
	  image = cv::imread(filenames[i+b], cv::IMREAD_UNCHANGED);	  
	  if (init) {
	    trt_yolox->initPreprocessBuffer(image.cols, image.rows);
	    init = false;
	  }
	  inputs.push_back(image);
	}
      }
      for (int b = 0; b < batch; b++) {	
	if (getRandomCrop()) {
	  cv::Mat src = getMultiScaleInference() ? inputs[0] : inputs[b];	  
	  roi.x = random(0, src.cols-100);
	  roi.y = random(0, src.rows-100);
	  roi.width = random(100, 500);
	  roi.height = random(100, 500);
	  roi.width = (roi.x + roi.width) > src.cols ? (src.cols-roi.x) : roi.width;
	  roi.height = (roi.y + roi.height) > src.rows ? (src.rows-roi.y) : roi.height;  	  
	}
	rois.push_back(roi);
      }

      tensorrt_yolox::ObjectArrays objects;
      if (rois[0].width !=-1 && rois[0].height != -1) {
	if (getMultiScaleInference()) {
	  trt_yolox->doMultiScaleInference(inputs[0], objects, rois);
	} else {
	  trt_yolox->doInferenceWithRoi(inputs, objects, rois);
	}
      } else {
	trt_yolox->doInference(inputs, objects);
      }
      
      int num_multitask = trt_yolox->getMultitaskNum();
      for (int b = 0; b < batch; b++) {
	cv::Mat src = getMultiScaleInference() ? inputs[0] : inputs[b];
	if (rois[b].width !=-1 && rois[b].height != -1) {
	  cv::rectangle(
			src, cv::Point(rois[b].x, rois[b].y), cv::Point(rois[b].x+rois[b].width, rois[b].y+rois[b].height), cv::Scalar(255, 255, 255), 1, 8, 0);
	}

	if (num_multitask) {
	  for (int m = 0 ; m < num_multitask; m++) {
	    auto cmask = trt_yolox->getColorizedMask(m, seg_cmap);	    
	    cv::Mat resized;
	    if (flg_save) {

	      fs::path p (filenames[i+b]);
	      std::string name = p.filename().string();
	      std::ostringstream sout;
	      p = save_path;
	      p.append("segmentations");
	      cv::resize(cmask, resized, cv::Size(src.cols, src.rows), 0, 0, cv::INTER_NEAREST);
	      if (!contain(name, ".png")) {
		replaceOtherStr(name, ".jpg", ".png");
	      }
	      save_image(resized, p.string(), name);
	    }      	    
	    if (!dont_show) {	    
	      cv::namedWindow("cmask", 0);
	      cv::imshow("cmask", cmask);
	      if (cv::waitKey(1) == 'q') break;
	      cv::resize(cmask, resized, cv::Size(src.cols, src.rows), 0, 0, cv::INTER_NEAREST);	      
	      cv::addWeighted(inputs[0], 1.0, resized, 0.5, 0.0, src);	      
	    }	  
	  }
	}
	
	for (const auto & object : objects[b]) {
	  const auto left = object.x_offset + rois[b].x;
	  const auto top = object.y_offset + rois[b].y;
	  const auto right = std::clamp(left + object.width, 0, src.cols);
	  const auto bottom = std::clamp(top + object.height, 0, src.rows);
	  if (colormap.size()) {
	    char buff[128];
	    std::vector<int> rgb = colormap[object.type];
	    cv::rectangle(
			  src, cv::Point(left, top), cv::Point(right, bottom), cv::Scalar(rgb[2], rgb[1], rgb[0]), 2);
	    sprintf(buff, "%2.0f%%", object.score * 100);
	    cv::putText(src, buff, cv::Point(left, top), 0, 0.5, cv::Scalar(rgb[2], rgb[1], rgb[0]), 2);
	  } else {
	  cv::rectangle(
			src, cv::Point(left, top), cv::Point(right, bottom), cv::Scalar(0, 0, 255), 3, 8, 0);
	  }
	}

	if (!dont_show) {
	  if (getMultiScaleInference()) {
	    if (b == (batch-1)) {
	      cv::imshow("multi-scale inference image" + std::to_string(b), src);
	    }
	  } else {
	    cv::namedWindow("inference image" + std::to_string(b), cv::WINDOW_NORMAL);
	    cv::imshow("inference image" + std::to_string(b), src);
	  }
	}
	if (dumpPath != "not-specified") {
	  fs::path p (filenames[i+b]);
	  std::string filename = p.filename().string();
	  std::vector<std::string> names = get_names();
	  write_prediction(dumpPath, filename, names, objects[b], src.cols, src.rows);
	}
	if (outputPath != "not-specified") {
	  fs::path p (filenames[i+b]);	  
	  std::string filename = p.filename().string();
	  std::vector<std::string> names = get_names();
	  write_label(outputPath, filename, names, objects[0], src.cols, src.rows);
	}      
	
	if (flg_save) {
	  fs::path p (filenames[i+b]);
	  std::string name = p.filename().string();	
	  std::ostringstream sout;
	  p = save_path;
	  p.append("detections");
	  save_image(src, p.string(), name);
	}      
      }
      if (!dont_show) {
	  int k = cv::waitKey(0);
	  if (k == 32) {
	    int b = 0;
	    fs::path p (filenames[i+b]);
	    std::string name = p.filename().string();
	    std::cout << "Save... " << name << std::endl;
	    cv::Mat src = cv::imread(filenames[i+b], cv::IMREAD_UNCHANGED);
	    save_image(src, "log/", name);
	    cv::waitKey(0);
	  }	

      }
    }
  } else if (video_path != "" || cam_id != -1) {
   std::cout << video_path << std::endl;
    cv::VideoCapture video;
    if (cam_id != -1) {
      video.open(cam_id);
    } else {
      video.open(video_path);
    }
    cv::Mat image;
    std::string window_name = "inference image";
    cv::namedWindow(window_name, 0);
    if (window_info.w !=0 && window_info.h !=0) {
      cv::resizeWindow(window_name, window_info.w, window_info.h);
    }
    cv::moveWindow(window_name, window_info.x, window_info.y);
    int frame_count = 0;
    while (1) {
      video >> image;
      if (image.empty() == true) break;
      if (init) {
	trt_yolox->initPreprocessBuffer(image.cols, image.rows);
	init = false;
      }      
      tensorrt_yolox::ObjectArrays objects;
      trt_yolox->doInference({image}, objects);

      int num_multitask = trt_yolox->getMultitaskNum();
      if (num_multitask) {
	for (int m = 0 ; m < num_multitask; m++) {
	  auto cmask = trt_yolox->getColorizedMask(m, seg_cmap);	  
	  cv::namedWindow("cmask", 0);
	  cv::imshow("cmask", cmask);
	  if (cv::waitKey(1) == 'q') break;
	  cv::Mat resized;
	  cv::resize(cmask, resized, cv::Size(image.cols, image.rows), 0, 0, cv::INTER_NEAREST);
	  cv::addWeighted(image, 1.0, resized, 0.5, 0.0, image);
	  if (num_multitask) {
	    for (int m = 0 ; m < num_multitask; m++) {
	      if (flg_save) {
		fs::path p;
		std::ostringstream sout;
		std::string name = "frame_" + sout.str() + ".jpg";	  		
		p = save_path;
		p.append("segmentations");
		replaceOtherStr(name, ".jpg", ".png");
		
		save_image(resized, p.string(), name);
	      }
	    }
	  }
	}
      }
      
      if (!dont_show) { 
	for (const auto & object : objects[0]) {
	  const auto left = object.x_offset;
	  const auto top = object.y_offset;
	  const auto right = std::clamp(left + object.width, 0, image.cols);
	  const auto bottom = std::clamp(top + object.height, 0, image.rows);
	  if (colormap.size()) {
	    char buff[128];
	    std::vector<int> rgb = colormap[object.type];
	    cv::rectangle(
			image, cv::Point(left, top), cv::Point(right, bottom), cv::Scalar(rgb[2], rgb[1], rgb[0]), 2);
	    sprintf(buff, "%2.0f%%", object.score * 100);
	    cv::putText(image, buff, cv::Point(left, top), 0, 0.5, cv::Scalar(rgb[2], rgb[1], rgb[0]), 2);
	  } else {
	    
	    cv::rectangle(
			  image, cv::Point(left, top), cv::Point(right, bottom), cv::Scalar(0, 0, 255), 3, 8, 2);
	  }
	  
	}	
	cv::imshow(window_name, image);

	if (flg_save) {	  
	  fs::path p;
	  std::ostringstream sout;
	  sout << std::setfill('0') << std::setw(6) << frame_count++;	  
	  std::string name = "frame_" + sout.str() + ".jpg";	  
	  p = save_path;
	  p.append("detections");
	  save_image(image, p.string(), name);
	}      	
	if (cv::waitKey(1) == 'q') break;	
      }      
    } 
  }  

  
  
  if (prof) {
    trt_yolox->printProfiling();
  }

  return 0;
}
