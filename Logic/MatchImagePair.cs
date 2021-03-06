﻿using Emgu.CV;
using Emgu.CV.Features2D;
using Emgu.CV.Structure;
using Emgu.CV.Util;
using System;
using System.Collections.Generic;
using System.Drawing;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Egomotion
{
    public class MatchingResult
    {
        VectorOfPointF lps;
        VectorOfPointF rps;

        public VectorOfPointF LeftPoints
        {
            get => lps;
            set
            {
                lps = value;
                LeftPointsList = lps.ToList();
            }
        }
        public VectorOfPointF RightPoints
        {
            get => rps;
            set
            {
                rps = value;
                RightPointsList = rps.ToList();
            }
        }

        public List<double> Distances { get; set; }

        public MKeyPoint[] LeftKps { get; set; }
        public MKeyPoint[] RightKps { get; set; }
        public VectorOfDMatch Matches { get; set; }
        public Mat LeftDescriptors { get; set; }
        public Mat RightDescriptors { get; set; }

        // For convience
        public List<PointF> LeftPointsList { get; private set; }
        public List<PointF> RightPointsList { get; private set; }
    }

    public static class MatchImagePair
    {
        public static void FindFeatures(Mat image, Feature2D detector, Feature2D descriptor, out MKeyPoint[] kps, out Mat descriptors)
        {
            kps = detector.Detect(image);
            VectorOfKeyPoint vectorOfKp = new VectorOfKeyPoint(kps);
            descriptors = new Mat();
            descriptor.Compute(image, vectorOfKp, descriptors);
            kps = vectorOfKp.ToArray();
        }

        public static VectorOfDMatch FindMatches(MKeyPoint[]  kps1, MKeyPoint[] kps2, Mat desc1, Mat desc2, DistanceType distanceType, double maxDistance)
        {
        //    var res = MatchClosePoints.Match(kps1, kps2, desc1, desc2, distanceType, maxDistance);
        //    return new VectorOfDMatch(res.ToArray());

            VectorOfDMatch matches = new VectorOfDMatch();
            BFMatcher bf = new BFMatcher(distanceType, true);
            bf.Match(desc1, desc2, matches);
            return matches;
        }

        public static void MacthesToPointLists(IEnumerable<MDMatch> sortedMatches, MKeyPoint[] kp1, MKeyPoint[] kp2,
            out VectorOfPointF leftPoints, out VectorOfPointF rightPoints, out List<double> distances)
        {
            leftPoints = new VectorOfPointF();
            rightPoints = new VectorOfPointF();
            distances = new List<double>();

            foreach(var m in sortedMatches)
            {
                leftPoints.Push(new PointF[] { kp1[m.QueryIdx].Point });
                rightPoints.Push(new PointF[] { kp2[m.TrainIdx].Point });
                distances.Add(m.Distance);
            }
        }

        public static MatchingResult Match(Mat left, Mat right, Feature2D detector, Feature2D descriptor, DistanceType distanceType, double maxDistance)
        {
            FindFeatures(left, detector, descriptor, out MKeyPoint[] kps1, out Mat desc1);
            FindFeatures(right, detector, descriptor, out MKeyPoint[] kps2, out Mat desc2);
            return Match(kps1, desc1, kps2, desc2, distanceType, maxDistance);
        }
        public static MatchingResult Match(MKeyPoint[] kps1, Mat desc1, MKeyPoint[] kps2, Mat desc2, DistanceType distanceType, double maxDistance)
        {
            var matches = FindMatches(kps1, kps2, desc1, desc2, distanceType, maxDistance);

            var sortedMatches = matches.ToArray().Where((x) =>
            {
                double dx = kps1[x.QueryIdx].Point.X - kps2[x.TrainIdx].Point.X;
                double dy = kps1[x.QueryIdx].Point.Y - kps2[x.TrainIdx].Point.Y;
                return dx * dx + dy * dy < maxDistance * maxDistance;
            }).OrderBy((x) => x.Distance);
            MacthesToPointLists(sortedMatches, kps1, kps2, out VectorOfPointF leftPoints, out VectorOfPointF rightPoints, out List<double> distances);

            return new MatchingResult()
            {
                LeftPoints = leftPoints,
                RightPoints = rightPoints,
                LeftKps = kps1,
                RightKps = kps2,
                Matches = new VectorOfDMatch(sortedMatches.ToArray()),
                Distances = distances,
                LeftDescriptors = desc1,
                RightDescriptors = desc2
            };
        }
    }
}
