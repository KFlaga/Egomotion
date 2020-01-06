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
    }
}
