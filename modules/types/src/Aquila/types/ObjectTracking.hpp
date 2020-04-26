#pragma once
#include "ObjectDetection.hpp"
#include <Aquila/types/SyncedMemory.hpp>
#include <boost/circular_buffer.hpp>
#include <opencv2/video.hpp>
namespace aq
{

    template <class T, class U>
    cv::Vec<T, 2> center(const cv::Rect_<U>& rect)
    {
        return cv::Vec<T, 2>(rect.x + rect.width / 2, rect.y + rect.height / 2);
    }

    template <class T, class U>
    float iou(const cv::Rect_<T>& r1, const cv::Rect_<U>& r2)
    {
        float intersection = (r1 & r2).area();
        float rect_union = (r1 | r2).area();
        return intersection / rect_union;
    }

    /*!
     * \brief extentDistance
     * \param meas1
     * \param meas2
     * \param dims
     * \return
     */
    template <int N>
    float extentDistance(const cv::Mat& measurement, const cv::Mat& state)
    {
        float value = 0.0f;
        // Position state, size state, position measurement, size measurement
        cv::Vec<float, N> Ps, Ss, Pm, Sm;
        for (int i = 0; i < N; ++i)
        {
            Ps[i] = state.at<float>(i);
            Ss[i] = state.at<float>(i + N * 2);
            Pm[i] = measurement.at<float>(i);
            Sm[i] = measurement.at<float>(i + N);
        }
        // Calculate a normalized metric such that the if the centroid of measurement
        // is within the bounds of state the score will be between 0 and 1
        //

        //             _______________
        //             |              |
        //             |              |
        //      _____________.Pm      | Score == 1
        //      |      |     |        |
        //      |      |______________|
        //      |   Ps .     |
        //      |            |
        //      |____________|_______________
        //                   |              |
        //                   |              |
        //                   |      Pm      | Score == 2
        //                   |              |
        //                   |_______________
        //  The score should be 0 if Ps and Ss lie ontop of each other
        for (int i = 0; i < N; ++i)
        {
            value += abs(Ps[i] - Pm[i]) / (Ss[i] / 2);
        }
        // Calculate a score based on the % change in size of the object
        for (int i = 0; i < N; ++i)
        {
            value += abs(Ss[i] - Sm[i]) / (Ss[i]);
        }
        // Normalize value by the number of dimensions
        value /= (float)N;
        return value;
    }

    struct AQUILA_EXPORTS TrackedObject2d
    {
        enum TrackingState
        {

        };

        enum
        {
            Dims = 2
        };
        typedef DetectedObject2d DetectionType;
        typedef std::shared_ptr<TrackedObject2d> Ptr;
        typedef std::vector<Ptr> TrackSet;
        TrackedObject2d() { detection_history.set_capacity(30); }
        /*!
         * \brief calculate a score wrt this track for association
         * \param obj object detection at current frame
         * \return confidence of matched detection
         */
        virtual float Score(const DetectedObject2d& obj) { return iou(obj.bounding_box, last_bb); }

        virtual void AddDetection(const DetectedObject2d& obj)
        {
            detection_history.push_back(obj);
            last_bb = obj.bounding_box;
            if (detection_history.size())
            {
                cv::Vec2f pos0 = center<float>(detection_history.back().bounding_box);
                cv::Vec2f pos1 = center<float>(obj.bounding_box);
                velocity = pos1 - pos0;
            }
        }

        virtual float Track(const aq::SyncedMemory& img, mo::Time ts, cv::cuda::Stream& stream)
        {
            // No tracking is done in base class
            return 0.0f;
        }

        /*!
         * \brief velocity is the current velocity estimation in image coordinates
         */
        cv::Vec2f velocity;
        /*!
         * \brief last_bb the last bounding box, either tracked or from the last detection
         */
        cv::Rect2f last_bb;
        /*!
         * \brief id unique track id
         */
        int id = 0;
        /*!
         * \brief initial_detection_timestamp is the timestamp of first detection
         */
        boost::optional<mo::Time> initial_detection_timestamp;
        /*!
         * \brief detection_history past N detections
         */
        boost::circular_buffer<DetectedObject2d> detection_history;
        TrackingState state;
    };
    typedef TrackedObject2d::TrackSet TrackSet2d;

    struct AQUILA_EXPORTS TrackedObject3d
    {
        enum
        {
            Dims = 3
        };
        typedef DetectedObject3d DetectionType;
        boost::circular_buffer<DetectedObject3d> detection_history;
    };

    /*!
     * \brief The KalmanTrackedObject2d struct is used to predict movement with a kalman filter
     *        This object will track a detected objects centroid and extent
     *  http://www.robot-home.it/blog/en/software/ball-tracker-con-filtro-di-kalman/
     */
    template <class T>
    struct AQUILA_EXPORTS KalmanTrackedObject : public T
    {
        KalmanTrackedObject(cv::Vec<float, T::Dims> Ep = cv::Vec<float, T::Dims>::all(1e-2),
                            cv::Vec<float, T::Dims> Ev = cv::Vec<float, T::Dims>::all(1),
                            cv::Vec<float, T::Dims> Es = cv::Vec<float, T::Dims>::all(1e-2))
            : initialized(false), T()
        {
            // The state of a 2d object is the centroid (x,y), velocity (x,y) and size (x,y) (6)
            // The state of a 3d object is the centroid (x,y, z), velocity (x,y,z) and size (x,y,z) (9)
            int state_size = T::Dims * 3;
            // The measurement at any given detection should be centroid and size
            int meas_size = T::Dims * 2;
            kf = cv::KalmanFilter(state_size, meas_size, 0, CV_32F);
            // dT is the timestep between consecutive measurements in seconds
            // [ 1 0 dT 0  0 0 ]
            // [ 0 1 0  dT 0 0 ]
            // [ 0 0 1  0  0 0 ]
            // [ 0 0 0  1  0 0 ]
            // [ 0 0 0  0  1 0 ]
            // [ 0 0 0  0  0 1 ]
            cv::setIdentity(kf.transitionMatrix);

            kf.measurementMatrix = cv::Mat::zeros(meas_size, state_size, CV_32F);
            // for 2d the measure matrix is:
            // [ 1 0 0 0 0 0 ]
            // [ 0 1 0 0 0 0 ]
            // [ 0 0 0 0 1 0 ]
            // [ 0 0 0 0 0 1 ]
            // for 3d the measure matrix is:
            // [ 1 0 0 0 0 0 0 0 0]
            // [ 0 1 0 0 0 0 0 0 0]
            // [ 0 0 1 0 0 0 0 0 0]
            // [ 0 0 0 0 0 0 1 0 0]
            // [ 0 0 0 0 0 0 0 1 0]
            // [ 0 0 0 0 0 0 0 0 1]
            for (int i = 0; i < T::Dims; ++i)
            {
                kf.measurementMatrix.at<float>(i, i) = 1;
                kf.measurementMatrix.at<float>(i + T::Dims, i + 2 * T::Dims) = 1;
            }

            // process Noise Covariance Matrix Q 2d
            // [ Ex 0  0    0 0    0 ]
            // [ 0  Ey 0    0 0    0 ]
            // [ 0  0  Ev_x 0 0    0 ]
            // [ 0  0  0    1 Ev_y 0 ]
            // [ 0  0  0    0 1    Ew ]
            // [ 0  0  0    0 0    Eh ]
            for (int i = 0; i < T::Dims; ++i)
            {
                kf.processNoiseCov.at<float>(i, i) = Ep[i];
                kf.processNoiseCov.at<float>(T::Dims + i, T::Dims + i) = Ev[i];
                kf.processNoiseCov.at<float>(2 * T::Dims + i, 2 * T::Dims + i) = Es[i];
            }
            cv::setIdentity(kf.measurementNoiseCov, cv::Scalar(1e-1));
        }

        float Score(const typename T::DetectionType& obj)
        {
            cv::Mat meas = measure(obj);
            float score = 0.0;
            if (obj.timestamp > predicted_timestamp)
            {
                Predict(*obj.timestamp);
            }
            else if (obj.timestamp < predicted_timestamp && obj.timestamp < current_state_timestamp)
            {
                return T::Score(obj);
            }
            if (obj.timestamp == predicted_timestamp)
            {
                // Score against predicted state
                // Position
                score = extentDistance<T::Dims>(meas, predicted_state);
                /*score += cv::norm( meas.rowRange(0, T::Dims),
                                   predicted_state.rowRange(0, T::Dims),
                                   cv::NORM_L2) / cv::norm(predicted_state.rowRange(0, T::Dims));
                // Size
                score += cv::norm( meas.rowRange(T::Dims, T::Dims * 2),
                                   predicted_state.rowRange(T::Dims * 2, T::Dims* 3),
                                   cv::NORM_L2) / cv::norm(predicted_state.rowRange(T::Dims, T::Dims * 2));*/
            }
            if (obj.timestamp == current_state_timestamp)
            {
                score = extentDistance<T::Dims>(meas, kf.statePost);
                // Position
                /*score += cv::norm(meas.rowRange(0, T::Dims),
                                  kf.statePost.rowRange(0, T::Dims),
                                  cv::NORM_L2) / cv::norm(kf.statePost.rowRange(0, T::Dims));
                // Size
                score += cv::norm( meas.rowRange(T::Dims, T::Dims * 2),
                                   kf.statePost.rowRange(T::Dims * 2, T::Dims* 3),
                                   cv::NORM_L2) / cv::norm(kf.statePost.rowRange(T::Dims, T::Dims * 3));*/
            }

            return score;
        }

        /*!
         * \brief When no detection is present in this frame, drift the track
         * \param timestamp timestamp of current frame
         */
        void Track(const aq::SyncedMemory& img, mo::Time ts)
        {
            // Can't drift a track to the past
            if (ts < current_state_timestamp)
                return;
            auto dt = ts - current_state_timestamp;
            for (int i = 0; i < T::Dims; ++i)
            {
                kf.transitionMatrix.at<float>(T::Dims + i, i) =
                    std::chrono::duration_cast<std::chrono::milliseconds>(dt) * 1000.0f;
            }
            cv::Mat state = kf.predict();
            kf.statePost = state;
            current_state_timestamp = ts;
            predicted_state = state;
            predicted_timestamp = ts;
        }

        void AddDetection(const typename T::DetectionType& obj)
        {
            cv::Mat meas = measure(obj);
            cv::Mat state(T::Dims * 3, 1, CV_32F);
            if (initialized)
            {
                auto dt = *obj.timestamp - current_state_timestamp;
                for (int i = 0; i < T::Dims; ++i)
                {
                    kf.transitionMatrix.at<float>(T::Dims + i, i) = dt.count();
                }
            }
            current_state_timestamp = *obj.timestamp;
            if (!initialized)
            {
                if (this->detection_history.size() > 0)
                {

                    // Initialize the internal kf state
                    for (int i = 0; i < T::Dims; ++i)
                    {
                        kf.errorCovPre.at<float>(i, i) = 1; // px
                        kf.errorCovPre.at<float>(T::Dims + i, T::Dims + i) = 1;
                        kf.errorCovPre.at<float>(2 * T::Dims + i, 2 * T::Dims + i) = 1; // px
                    }
                    cv::Mat prev_meas = measure(this->detection_history.back());
                    for (int i = 0; i < T::Dims; ++i)
                    {
                        // Position is set to current position
                        state.at<float>(i) = meas.at<float>(i);
                        // Velocity wrt previous detection
                        state.at<float>(i + T::Dims) = meas.at<float>(i) - prev_meas.at<float>(i);
                        // bb size
                        state.at<float>(i + 2 * T::Dims) = meas.at<float>(i + T::Dims);
                    }
                    kf.statePost = state;
                    initialized = true;
                    update(*this, state);
                }
            }
            else
            {
                // kf correct returns the corrected state
                update(*this, kf.correct(meas));
            }
            if (this->detection_history.size() == 0)
            {
                this->initial_detection_timestamp = obj.timestamp;
            }
            this->detection_history.push_back(obj);
        }

        // TODO probably need to do some timestamp adjustment here for non constant frame rate
        cv::Mat Predict(mo::Time ts)
        {
            CV_Assert(initialized);
            if (ts == current_state_timestamp)
                return kf.statePost;
            if (ts == predicted_timestamp)
                return predicted_state;

            auto dt = ts - current_state_timestamp;
            for (int i = 0; i < T::Dims; ++i)
            {
                kf.transitionMatrix.at<float>(T::Dims + i, i) =
                    std::chrono::duration_cast<std::chrono::milliseconds>(dt).count() * 1000.0f;
            }

            predicted_state = kf.predict();
            predicted_timestamp = ts;
            update(*this, predicted_state);
            return predicted_state;
        }

      protected:
        cv::Mat measure(const DetectedObject2d& obj)
        {
            cv::Mat meas(4, 1, CV_32F);
            meas.at<float>(0) = obj.bounding_box.x + obj.bounding_box.width / 2;
            meas.at<float>(1) = obj.bounding_box.y + obj.bounding_box.height / 2;
            meas.at<float>(2) = obj.bounding_box.width;
            meas.at<float>(3) = obj.bounding_box.height;
            return meas;
        }

        cv::Mat measure(const DetectedObject3d& obj)
        {
            cv::Mat meas(6, 1, CV_32F);
            auto t = obj.pose.translation();
            meas.at<float>(0) = t(0);
            meas.at<float>(1) = t(1);
            meas.at<float>(2) = t(2);
            auto S = obj.size;
            meas.at<float>(3) = S(0);
            meas.at<float>(4) = S(1);
            meas.at<float>(5) = S(2);
            return meas;
        }
        cv::Vec<float, T::Dims> velocity(const cv::Mat& state)
        {
            cv::Vec<float, T::Dims> output;
            for (int i = 0; i < T::Dims; ++i)
            {
                output.val[i] = state.at<float>(T::Dims + i);
            }
            return output;
        }
        cv::Vec<float, T::Dims> position(const cv::Mat& state)
        {
            cv::Vec<float, T::Dims> output;
            for (int i = 0; i < T::Dims; ++i)
            {
                output.val[i] = state.at<float>(i);
            }
            return output;
        }
        cv::Vec<float, T::Dims> size(const cv::Mat& state)
        {
            cv::Vec<float, T::Dims> output;
            for (int i = 0; i < T::Dims; ++i)
            {
                output.val[i] = state.at<float>(2 * T::Dims + i);
            }
            return output;
        }

        void update(TrackedObject3d& obj, const cv::Mat& state) {}

        void update(TrackedObject2d& obj, const cv::Mat& state)
        {
            float x = state.at<float>(0);
            float y = state.at<float>(1);
            float w = state.at<float>(4);
            float h = state.at<float>(5);
            obj.last_bb.x = x - w / 2;
            obj.last_bb.y = y - h / 2;
            obj.last_bb.height = h;
            obj.last_bb.width = w;
            obj.velocity = state.at<float>(2);
            obj.velocity = state.at<float>(3);
        }

        cv::KalmanFilter kf;
        bool initialized;
        mo::Time current_state_timestamp;
        mo::Time predicted_timestamp;
        cv::Mat predicted_state;
    };

    typedef KalmanTrackedObject<TrackedObject2d> KalmanTracked2d;
    typedef KalmanTrackedObject<TrackedObject3d> KalmanTracked3d;
}
