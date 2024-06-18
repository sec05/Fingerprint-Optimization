namespace NLA
{

    class Vector
    {
    public:
        Vector(double*, int);
        Vector(int);
        ~Vector();

        int dimension;
        double *components;

        void scale(double);
        void add(Vector*);
        void subtract(Vector*);
        double dot(Vector*);
        void makeUnitVector();
    };
}