pub mod combine {
    use std::cmp::PartialOrd;
    use std::iter::Peekable;
    use std::marker::PhantomData;

    #[derive(Clone)]
    pub struct CombineIterDescending<T: PartialOrd, I1: Iterator<Item = T>, I2: Iterator<Item = T>>
    where T: PartialOrd, I1: Iterator<Item = T> + Clone, I2: Iterator<Item = T>
    {
        p1: Peekable<I1>,
        p2: Peekable<I2>,
        _phantom: PhantomData<T>,
    }

    impl <T, I1, I2> CombineIterDescending<T, I1, I2> 
    where T: PartialOrd, I1: Iterator<Item = T> + Clone, I2: Iterator<Item = T>
    {
        pub fn new(i1: I1, i2: I2) -> Self {
            return CombineIterDescending {
                p1: i1.peekable(),
                p2: i2.peekable(),
                _phantom: Default::default(),
            }
        }
    }

    impl <T, I1, I2> Iterator for CombineIterDescending<T, I1, I2> 
    where T: PartialOrd, I1: Iterator<Item = T> + Clone, I2: Iterator<Item = T>
    {
        type Item = T;

        fn next(&mut self) -> Option<T> {
            return match self.p1.peek() {
                Some(v1) => {
                    match self.p2.peek() {
                        Some(v2) => {
                            if v1.gt(v2) {
                                self.p1.next()
                            } else {
                                self.p2.next()
                            }
                        }
                        None => self.p1.next()
                    }
                }
                None => self.p2.next()
            }
        }
    }
 
}
