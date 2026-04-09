#pragma once
#ifndef _VBOG_VBOG_H_
#define _VBOG_VBOG_H_

// SYSTEM INCLUDES
#include <list>                 // std::list is a linked list
#include <map>                  // std::map
#include <585/common/types.h>
#include <585/vbow/vbow.h>


// C++ PROJECT INCLUDES


namespace ivc
{
namespace student
{

    // ----------------------------- REQUIRED BY ALL STUDENTS -------------------------------------
    // vocabulary. This class needs to keep track of each unique element it is presented with
    // along with the idx of the element in the output feature vector.
    class Vocab
    {
    public:
        Vocab();

        void add(const ivc::GrayscaleByteImg& e);
        const size_t size() const;
        std::list<ivc::GrayscaleByteImg> ordered_elements() const;

    protected:
        std::map<ivc::GrayscaleByteImg, uint64_t, GrayscaleByteImg_comp_t>  _patch_to_idx;
        std::map<uint64_t, ivc::GrayscaleByteImg>                           _idx_to_patch;
    };

    // bag of words base class
    class BagOfWords
    {
    public:
        BagOfWords(const ivc::ByteDataset& X,
                   const size_t patch_size);    // element dimension (square). Will need to pad

        // takes a dataset of images in (one image per row) and returns the bow vector per image
        // virtual ... = 0; is how you make an abstract method in c++
        virtual ivc::FloatDataset transform(const ivc::ByteDataset& X) const = 0;

    protected:
        Vocab         _vocab;
        const size_t  _patch_size;
    };

    // binary bow....feature vector is binary (is element present or not in the img?)
    class BinaryBagOfWords : public BagOfWords
    {
    public:
        BinaryBagOfWords(const ivc::ByteDataset& X,
                         const size_t patch_size);

        ivc::FloatDataset transform(const ivc::ByteDataset& X) const;
    };

    // counting bow...feature vector is ints (how many times is element present in img?)
    class CountingBagOfWords : public BagOfWords
    {
    public:
      CountingBagOfWords(const ivc::ByteDataset& X,
                         const size_t patch_size);

      ivc::FloatDataset transform(const ivc::ByteDataset& X) const;
    };

    // one-vs-rest classifier...uses a binary classifier (e.g. logistic-regression in 585/vbow/vbow.h)
    // to handle multi-class classification problems.
    // one-vs-rest means that this model has *multiple* binary classifiers inside it: one classifier
    // per class in the dataset. The purpose of each binary classifier is to ask if the sample belongs
    // to a specific class (binary label: yes or no). You should use model confidence to assign final class
    // and break ties for the most-probable class arbitrarily.
    class OVR
    {
    public:
        OVR();

        // output are class labels not probabilities
        ivc::ProbVector predict(const ivc::FloatDataset& X) const;
        float_t cost(const ivc::FloatDataset& X,
                     const ivc::ProbVector& y_gt) const;

        void train(const ivc::FloatDataset& X,
                   const ivc::ProbVector& y_gt,
                   const float_t lr,
                   const size_t max_epochs);
    private:

    };

    // ---------------------- REQUIRED BY GRAD / BONUS FOR UNDERGRAD ------------------------------
    // function to break each image in a dataset up into multiple smaller "tiles"
    // the level is the level of tiling you should do.
    // levels:
    //
    //      0 -> no tiling
    //      1 -> break image up into quadrants
    //      2 -> break each quadrant into quadrants
    //  won't ask you to worry about other tiling levels
    ivc::ByteDataset tile_dataset(const ivc::ByteDataset& X,
                                  const size_t level);

    // histogram of gradients. You will need to calculate the gradient angles of each image
    // and bin them into 45 degree increments (please use degrees)
    // use the 3x3 sobel angle operator in <585/grad/grad.h> to get your gradient angles.
    class HistogramOfGradients
    {
    public:
        HistogramOfGradients();

        // assume the input dataset has already been tiled if need be
        ivc::FloatDataset transform(const ivc::ByteDataset& X) const;
    };

    // ---------------------- BONUS FOR GRAD / NO CREDIT FOR UNDERGRAD -----------------------------
    typedef enum balance_type
    {
        OVERSAMPLE = 0,
        UNDERSAMPLE = 1,
        SMOTE = 2
    } balance_type_t;

    // class balancing function
    // SMOTE stands for "Synthetic Minority Over-sampling TEchnique"
    std::tuple<ivc::ByteDataset,
               ivc::ProbVector> balance(const ivc::ByteDataset& X,
                                        const ivc::ProbVector& y_gt,
                                        const ivc::student::balance_type_t balance_type);
    

} // end of namespace student
} // end of namespace ivc


#endif // end of _VBOG_VBOG_H_

