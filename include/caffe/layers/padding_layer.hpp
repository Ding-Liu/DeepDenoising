#ifndef CAFFE_PADDING_LAYER_HPP_
#define CAFFE_PADDING_LAYER_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

//#include "caffe/layers/conv_layer.hpp"

namespace caffe {

/**
 * @brief Padding layer
 */

template <typename Dtype>
class PaddingLayer : public Layer<Dtype> {
public:
    explicit PaddingLayer(const LayerParameter& param)
        : Layer<Dtype>(param) {}
    virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
                            const vector<Blob<Dtype>*>& top);
    virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
                         const vector<Blob<Dtype>*>& top);

    virtual inline const char* type() const { return "Padding"; }
    virtual inline int ExactNumBottomBlobs() const { return 1; }
    virtual inline int ExactNumTopBlobs() const { return 1; }

protected:
    virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
                             const vector<Blob<Dtype>*>& top);
       virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
                                const vector<Blob<Dtype>*>& top);
    virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
                              const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
       virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
                                 const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);

    // data members
    // convolution parameter
//        LayerParameter bicubic_conv_param;

    // channels of output blob
    int pad_beg_;
    int pad_end_;
    bool pad_pos_;
    
    int num_;
    int channels_;
    int height_in_;
    int width_in_;
    int height_out_;
    int width_out_;

};  // class PaddingLayer

}  // namespace caffe

#endif  // CAFFE_PADDING_LAYER_HPP_