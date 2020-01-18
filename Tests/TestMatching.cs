using System;
using System.Text;
using System.Collections.Generic;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using Egomotion;
using Emgu.CV;
using Emgu.CV.Structure;
using System.Drawing;

namespace Tests
{
    [TestClass]
    public class TestMatching
    {
        public TestMatching()
        {
        }
        
        [TestMethod]
        public void TestMatchClosePoints()
        {
            Image<Arthmetic, double> desc1 = new Image<Arthmetic, double>(new double[,,] {
                { {2}, {2}, } ,
                { {1}, {0}, } ,
                { {2}, {0}, } ,
                { {0}, {1}, } ,
                { {1}, {1}, } ,
                { {0}, {2}, } ,
            });

            Image<Arthmetic, double> desc2 = new Image<Arthmetic, double>(new double[,,] {
                { {1}, {0}, } ,
                { {1}, {1}, } ,
                { {0}, {1}, } ,
                { {2}, {0}, } ,
                { {2}, {2}, } ,
                { {0}, {2}, } ,
            });

            MKeyPoint[] kps1 = new MKeyPoint[]
            {
                new MKeyPoint() { Point = new PointF(15, 0) },
                new MKeyPoint() { Point = new PointF(0, 0) },
                new MKeyPoint() { Point = new PointF(6, -5) },
                new MKeyPoint() { Point = new PointF(3, 0) },
                new MKeyPoint() { Point = new PointF(12, 0) },
                new MKeyPoint() { Point = new PointF(9, 0) },
            };

            MKeyPoint[] kps2 = new MKeyPoint[]
            {
                new MKeyPoint() { Point = new PointF(0, 2) },
                new MKeyPoint() { Point = new PointF(9, 0) },
                new MKeyPoint() { Point = new PointF(3, 5) },
                new MKeyPoint() { Point = new PointF(6, 7) },
                new MKeyPoint() { Point = new PointF(15, 0) },
                new MKeyPoint() { Point = new PointF(9, 7) },
            };

            var matches = MatchClosePoints.Match(kps1, kps2, desc1.Mat, desc2.Mat, Emgu.CV.Features2D.DistanceType.L2, 5.5, true);
        }

        [TestMethod]
        public void TestThreeViewsCorrespondences()
        {
            MKeyPoint[] kps1 = new MKeyPoint[]
            {
                new MKeyPoint() { Point = new PointF(0, 0) },
                new MKeyPoint() { Point = new PointF(5, 2) },
                new MKeyPoint() { Point = new PointF(5, 7) },
                new MKeyPoint() { Point = new PointF(2, 1) },
                new MKeyPoint() { Point = new PointF(-4, 2) },
            };

            MKeyPoint[] kps2a = new MKeyPoint[]
            {
                new MKeyPoint() { Point = new PointF(0, 2) },
                new MKeyPoint() { Point = new PointF(9, 0) },
                new MKeyPoint() { Point = new PointF(9, -4) },
                new MKeyPoint() { Point = new PointF(6, 1) },
                new MKeyPoint() { Point = new PointF(-3, -3) },
            };

            MKeyPoint[] kps2b = new MKeyPoint[]
            {
                new MKeyPoint() { Point = new PointF(0, 2) },
                new MKeyPoint() { Point = new PointF(9, 0) },
                new MKeyPoint() { Point = new PointF(9, -4) },
                new MKeyPoint() { Point = new PointF(6, 1) },
                new MKeyPoint() { Point = new PointF(-3, -3) },
            };

            MKeyPoint[] kps3 = new MKeyPoint[]
            {
                new MKeyPoint() { Point = new PointF(0, 4) },
                new MKeyPoint() { Point = new PointF(11, 2) },
                new MKeyPoint() { Point = new PointF(11, -6) },
                new MKeyPoint() { Point = new PointF(8, 1) },
                new MKeyPoint() { Point = new PointF(-7, 0) },
            };

            MDMatch[] matches12 = new MDMatch[4]
            {
                new MDMatch { QueryIdx = 1, TrainIdx = 0 },
                new MDMatch { QueryIdx = 0, TrainIdx = 3 },
                new MDMatch { QueryIdx = 2, TrainIdx = 1 },
                new MDMatch { QueryIdx = 4, TrainIdx = 2 },
            };

            MDMatch[] matches23 = new MDMatch[4]
            {
                new MDMatch { QueryIdx = 0, TrainIdx = 4 },
                new MDMatch { QueryIdx = 1, TrainIdx = 2 },
                new MDMatch { QueryIdx = 4, TrainIdx = 0 },
                new MDMatch { QueryIdx = 2, TrainIdx = 1 },
            };

            MatchingResult match12 = new MatchingResult()
            {
                LeftKps = kps1,
                RightKps = kps2a,
                Matches = new Emgu.CV.Util.VectorOfDMatch(matches12)
            };
            MatchingResult match23 = new MatchingResult()
            {
                LeftKps = kps2b,
                RightKps = kps3,
                Matches = new Emgu.CV.Util.VectorOfDMatch(matches23)
            };

            List<Correspondences.MatchPair> correspondences = Correspondences.FindCorrespondences12to23(match12, match23);

            Assert.AreEqual(3, correspondences.Count);
            // They are sorted same as matches23
            var c = correspondences[0];
            Assert.AreEqual(1, c.Match12.QueryIdx);
            Assert.AreEqual(0, c.Match12.TrainIdx);
            Assert.AreEqual(0, c.Match23.QueryIdx);
            Assert.AreEqual(4, c.Match23.TrainIdx);

            Assert.AreEqual(kps1[1].Point, c.Kp1.Point);
            Assert.AreEqual(kps2a[0].Point, c.Kp2.Point);
            Assert.AreEqual(kps3[4].Point, c.Kp3.Point);

            c = correspondences[1];
            Assert.AreEqual(2, c.Match12.QueryIdx);
            Assert.AreEqual(1, c.Match12.TrainIdx);
            Assert.AreEqual(1, c.Match23.QueryIdx);
            Assert.AreEqual(2, c.Match23.TrainIdx);

            c = correspondences[2];
            Assert.AreEqual(4, c.Match12.QueryIdx);
            Assert.AreEqual(2, c.Match12.TrainIdx);
            Assert.AreEqual(2, c.Match23.QueryIdx);
            Assert.AreEqual(1, c.Match23.TrainIdx);
        }
    }
}
