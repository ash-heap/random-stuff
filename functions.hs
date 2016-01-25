import Data.List
import Data.Monoid
import Data.Ratio
import System.IO
import System.Random
import Control.Monad
import Control.Applicative
import Control.Monad.State
import qualified Control.Monad.Writer as W
import qualified Data.Foldable as F
import qualified Data.Map as Map
import qualified Data.Set as Set
import qualified Geometry.Sphere as Sphere  
import qualified Geometry.Cuboid as Cuboid  
import qualified Geometry.Cube as Cube  


--------------------------------------------------------------------------------
-- Starting Out
--------------------------------------------------------------------------------

doubleUs x y = doubleMe x + doubleMe y

doubleMe x = x + x

doubleSmallNumber x = if x > 100 then x else x*2

doubleSmallNumber' x = (if x > 100 then x else x*2) + 1

conanO'Brien = "It's a-me, Conan O'Brien!"

boomBangs xs = [if x < 10 then "BOOM!" else "BANG!" | x <- xs, odd x]

--length' xs = sum [1 | _ <- xs]

removeNonUppercase :: String -> String -- optional type declaration of function.
--removeNonUppercase :: [Char] -> [Char] -- alternate form.
removeNonUppercase st = [c | c <- st, c `elem` ['A'..'Z']]

triangles = [ (a,b,c) | c <- [1..10], b <- [1..10], a <- [1..10]]

rightTriangles = [ (a,b,c) | c <- [1..10], b <- [1..c], a <- [1..b], a^2 + b^2 == c^2]

rightTriangles' = [ (a,b,c) | c <- [1..10], b <- [1..c], a <- [1..b], a^2 + b^2 == c^2, a + b + c == 24]




--------------------------------------------------------------------------------
-- Types and Typeclasses
--------------------------------------------------------------------------------

factorial :: Integer -> Integer
factorial n = product [1..n]

circumference :: Float -> Float
circumference r = 2 * pi * r


circumference' :: Double -> Double
circumference' r = 2 * pi * r

length' xs = fromIntegral (length xs)





--------------------------------------------------------------------------------
-- Syntax in Functions
--------------------------------------------------------------------------------

lucky :: (Integral a) => a -> String
lucky 7 = "LUCKY NUMBER SEVEN!"
lucky x = "Sorry, you're out of luck, pal!"

sayMe :: (Integral a) => a -> String
sayMe 1 = "One!"
sayMe 2 = "Two!"
sayMe 3 = "Three!"
sayMe 4 = "Four!"
sayMe 5 = "Five!"
sayMe _ = "Not between 1 and 5"

factorial' :: (Integral a) => a -> a
factorial' 0 = 1
factorial' n = n * factorial' (n - 1)

charName :: Char -> String
charName 'a' = "Albert"
charName 'b' = "Broseph"
charName 'c' = "Cecil"

addVectors :: (Num a) => (a, a) -> (a, a) -> (a, a)
addVectors (x1, y1) (x2, y2) = (x1 + x2, y1 + y2)

first :: (a, b, c) -> a
first (x, _, _) = x

second :: (a, b, c) -> b
second (_, y, _) = y

third :: (a, b, c) -> c
third (_, _, z) = z

head' :: [a] -> a
head' [] = error "Can't call head on an empty list, dummy!"
head' (x:_) = x

tell :: (Show a) => [a] -> String
tell []       = "The list is empty"
tell (x:[])   = "The list has one element: " ++ show x
tell (x:y:[]) = "The list has two elements: " ++ show x ++ " and " ++ show y
tell (x:y:_)  = "The list is long. The first two elements are: " ++ show x ++ " and " ++ show y

length'' :: (Num b) => [a] -> b
length'' [] = 0
length'' (_:xs) = 1 + length'' xs

sum' :: (Num a) => [a] -> a
sum' [] = 0
sum' (x:xs) = x + sum' xs

capital :: String -> String
capital "" = "Empty string, whoops!"
capital all@(x:xs) = "The first letter of " ++ all ++ " is " ++ [x]

bmi :: (RealFloat a) => a -> a -> a
bmi w h = w/(h * h) * 703

bmiTell :: (RealFloat a) => a -> a -> String
bmiTell w h
    | bmiVal <= skinny = "You're underweight, you emo, you!"
    | bmiVal <= normal = "You're supposedly normal. I bet you're ugly!"
    | bmiVal <= fat    = "You're fat! Lose some weight, fatty!"
    | otherwise        = "You're a whale, congratulations!"
    where bmiVal = bmi w h
          skinny = 18.5
          normal = 25.0
          fat = 30.0

max' :: (Ord a) => a -> a -> a
max' a b
    | a > b     = a
    | otherwise = b

compare' :: (Ord a) => a -> a -> Ordering
a `compare'` b
    | a > b     = GT
    | a == b    = EQ
    | otherwise = LT

initials :: String -> String -> String
initials firstname lastname = [f] ++ ". " ++ [l] ++ "."
    where (f:_) = firstname
          (l:_) = lastname

calcBmis :: (RealFloat a) => [(a, a)] -> [a]
calcBmis xs = [bmi w h | (w, h) <- xs]
    where bmi w h = w/(h * h) * 703

cylinder :: (RealFloat a) => a -> a -> a
cylinder r h = 
    let sideArea = 2 * pi * r * h
        topArea = pi * r^2
    in  sideArea + 2 * topArea

calcBmis' :: (RealFloat a) => [(a, a)] -> [a]
calcBmis' xs = [bmi | (w, h) <- xs, let bmi = (w/(h^2)) * 703]

describeList :: [a] -> String
describeList xs = "The list is " ++ case xs of
    []  -> "empty"
    [_] -> "a singleton list"
    _   -> "a longer list"

describeList' :: [a] -> String
describeList' xs = "The list is " ++ what xs
    where what []  = "empty"
          what [_] = "a singleton list"
          what _   = "a longer list"

ack :: (Integral a) => a -> a -> a
ack m n
    | m == 0            = n + 1
    | m > 0 && n == 0   = ack (m-1) 1
    | m > 0 && n > 0    = ack (m-1) (ack m (n-1))
    | otherwise = error "you are an idiot!"



--------------------------------------------------------------------------------
-- Recursion
--------------------------------------------------------------------------------

fib :: (Integral a) => a -> a
fib 0 = 0
fib 1 = 1
fib n = (fib (n - 1)) + (fib (n - 2))

maximum' :: (Ord a) => [a] -> a
maximum' []  = error "maximum of empty list"
maximum' [x] = x
maximum' (x:xs)
    | x > maxTail = x
    | otherwise   = maxTail
    where maxTail = maximum' xs

maximum'' :: (Ord a) => [a] -> a
maximum'' []  = error "maximum of empty list"
maximum'' [x] = x
maximum'' (x:xs) = max x (maximum' xs)

replecate' ::(Num i, Ord i) => i -> a -> [a]
replecate' n x
    | n <= 0    = []
    | otherwise = x:replecate' (n - 1) x

take' :: (Num i, Ord i) => i -> [a] -> [a]
take' n _
    | n <= 0    = []
take' _ []      = []
take' n (x:xs) = x:take' (n - 1) xs

reverse' :: [a] -> [a]
reverse' [] = []
reverse' (x:xs) = reverse' xs ++ [x]

repeat' :: a -> [a]
repeat' x = x:repeat' x

zip' :: [a] -> [b] -> [(a,b)]
zip' _ [] = []
zip' [] _ = []
zip' (x:xs) (y:ys) = (x,y):zip' xs ys

elem' :: (Eq a) => a -> [a] -> Bool
elem' a [] = False
elem' a (x:xs)
    | a == x    = True
    | otherwise = a `elem'` xs

quicksort :: (Ord a) => [a] -> [a]
quicksort [] = []
quicksort (x:xs) = smallerSorted ++ [x] ++ biggerSorted
    where smallerSorted = quicksort [a | a <- xs, a <= x]
          biggerSorted  = quicksort [a | a <- xs, a > x]



--------------------------------------------------------------------------------
-- Higher Order Functions
--------------------------------------------------------------------------------

multThree :: (Num a) => a -> a -> a -> a
multThree x y z = x * y * z

divideByTen :: (Floating a) => a -> a
divideByTen = (/10)

isUpperAlphanum :: Char -> Bool
isUpperAlphanum = (`elem` ['A'..'Z'])

isLowerAlphanum :: Char -> Bool
isLowerAlphanum = (`elem` ['a'..'z'])

isAlphanum :: Char -> Bool
isAlphanum c = (isUpperAlphanum c) || (isLowerAlphanum c)

applyTwice :: (a -> a) -> a -> a
applyTwice f x = f (f x)

zipWith' :: (a -> b -> c) -> [a] -> [b] -> [c]
zipWith' _ [] _ = []
zipWith' _ _ [] = []
zipWith' f (x:xs) (y:ys) = f x y : zipWith' f xs ys

-- flip' :: (a -> b -> c) -> (b -> a -> c)
-- flip' f x y = f y x

flip' :: (a -> b -> c) -> (b -> a -> c)
flip' f = \x y -> f y x

quicksort' :: (Ord a) => [a] -> [a]
quicksort' [] = []
quicksort' (x:xs) = smallerSorted ++ [x] ++ biggerSorted
    where smallerSorted = quicksort . filter (<=x) $ xs
          biggerSorted = quicksort . filter (>x) $ xs

largestDivisible :: (Integral a) => a
largestDivisible = head . filter (\x -> x `mod` 3829 == 0) $ [100000,99999..]

chain :: (Integral a) => a -> [a]
chain 1 = [1]
chain n
    | even n = n:chain (n `div` 2)
    | odd n  = n:chain (n*3 + 1)

numLongChains :: Int
numLongChains = length . filter (\xs -> (length xs) > 15) . map chain $ [1..100]

addThree :: (Num a) => a -> a -> a -> a
addThree x y z = x + y + z

addThree' :: (Num a) => a -> a -> a -> a
addThree' = \x -> \y -> \z -> x + y + z

sum'' :: (Num a) => [a] -> a
sum'' xs = foldl (+) 0 xs

elem'' :: (Eq a) => a -> [a] -> Bool
elem'' y ys = foldl (\acc x -> x == y || acc) False ys

map' :: (a -> b) -> [a] -> [b]
map' f xs = foldr (\x acc -> f x : acc) [] xs

maximum''' :: (Ord a) => [a] -> a
maximum''' = foldl1 (\acc x -> if x > acc then x else acc)

reverse'' :: [a] -> [a]
reverse'' = foldl (flip (:)) []

product' :: (Num a) => [a] -> a
product' = foldr1 (*)

filter' :: (a -> Bool) -> [a] -> [a]
filter' p = foldl (\acc x -> if p x then x : acc else acc) []

head'' :: [a] -> a
head'' = foldr1 (\x _ -> x)

last' :: [a] -> a
last' = foldl1 (\_ x -> x)

sqrtSums :: Int
sqrtSums = (1+) . length . takeWhile (<1000) . scanl1 (+) . map sqrt $ [1..]
-- sqrtSums = length (takeWhile (<1000) (scanl1 (+) (map sqrt [1..]))) + 1

sum''' :: (Num a) => [a] -> a
sum''' = foldl (+) 0 -- point free form

fn :: (RealFloat a, Integral b) => a -> b
fn = ceiling . negate . tan . cos . max 50 -- point free form
-- fn x = ceiling (negate (tan (cos (max 50 x))))

oddSquareSum :: Integer
oddSquareSum = sum . takeWhile (<10000) . filter odd . map (^2) $ [1..]
-- oddSquareSum = sum (takeWhile (<10000) (filter odd (map (^2) [1..])))



--------------------------------------------------------------------------------
-- Modules
--------------------------------------------------------------------------------

numUniques :: (Eq a, Num b) => [a] -> b
numUniques = fromIntegral . length . nub

search :: (Eq a) => [a] -> [a] -> Bool
search needle haystack = foldl searchHere False (tails haystack)
    where nlen = length needle
          searchHere acc x = take nlen x == needle || acc

quicksort'' :: (Ord a) => [a] -> [a]
quicksort'' [] = []
quicksort'' (x:xs) = quicksort'' smaller ++ [x] ++ quicksort'' bigger
    where (smaller, bigger) = partition (<=x) xs

fromList' :: (Ord k) => [(k, v)] -> Map.Map k v
fromList' = foldr (\(k,v) acc -> Map.insert k v acc) Map.empty



--------------------------------------------------------------------------------
-- Making Our Own Types and Typeclasses
--------------------------------------------------------------------------------

data Point = Point Float Float deriving (Show)
data Shape = Circle Point Float | Rectangle Point Point deriving (Show)

surface :: Shape -> Float
surface (Circle _ r) = pi * r ^ 2
surface (Rectangle (Point x1 y1) (Point x2 y2)) = (abs $ x2 - x1) * (abs $ y2 - y1)

nudge :: Shape -> Float -> Float -> Shape
nudge (Circle (Point x y) r) a b = Circle (Point (x + a) (y + b)) r
nudge (Rectangle (Point x1 y1) (Point x2 y2)) a b 
    = Rectangle (Point (x1 + a) (y1 + b)) (Point (x2 + a) (y2 + b))

baseCircle :: Float -> Shape
baseCircle r = Circle (Point 0 0) r

baseRect :: Float -> Float -> Shape
baseRect width height = Rectangle (Point 0 0) (Point width height)

data Person = Person 
    { firstName :: String 
    , lastName :: String 
    , age :: Int 
    } deriving (Eq, Show, Read)

data Car = Car 
    { company :: String 
    , model :: String 
    , year :: Int 
    } deriving (Show)

tellCar :: Car -> String
tellCar (Car {company = c, model = m, year = y}) 
    = "This " ++ c ++ " " ++ m ++ " was made in " ++ show y

data Vector a = Vector a a a deriving (Show)

vplus :: (Num t) => Vector t -> Vector t -> Vector t
(Vector i j k) `vplus` (Vector l m n) = Vector (i + l) (j + m) (k + n)

vectMult :: (Num t) => Vector t -> t -> Vector t
(Vector i j k) `vectMult` m = Vector (i*m) (j*m) (k*m)

scalarMult :: (Num t) => Vector t -> Vector t -> t
(Vector i j k) `scalarMult` (Vector l m n) = i*l + j*m + k*n

data Day = Monday | Tuesday | Wednesday | Thursday | Friday | Saturday | Sunday
    deriving (Eq, Ord, Show, Read, Bounded, Enum)

data LockerState = Taken | Free deriving (Show, Eq)

type Code = String

type LockerMap = Map.Map Int (LockerState, Code)

lockerLookup :: Int -> LockerMap -> Either String Code
lockerLookup lockerNumber map = 
    case Map.lookup lockerNumber map of
        Nothing -> Left $ "Locker number " ++ show lockerNumber ++ " doesn't exist!"
        Just (state, code) -> case state of
            Free -> Right code
            Taken -> Left $ "Locker " ++ show lockerNumber ++ " is already taken!"


lockers :: LockerMap  
lockers = Map.fromList   
    [(100,(Taken,"ZD39I"))  
    ,(101,(Free,"JAH3I"))  
    ,(103,(Free,"IQSA9"))  
    ,(105,(Free,"QOTSA"))  
    ,(109,(Taken,"893JJ"))  
    ,(110,(Taken,"99292"))  
    ]  

infixr 5 :-:
data List a = Empty | a :-: (List a) deriving (Show, Read, Eq, Ord)

-- infixr 5 .++
-- (.++) :: List a -> List a -> List a
-- Empty .++ ys = ys
-- (x :-: xs) .++ ys = x :-: (xs .++ ys)

infixr 5 .++
(.++) :: List a -> List a -> List a
(.++) Empty ys = ys
(.++) (x :-: xs) ys = x :-: (xs .++ ys)

data Tree a = Leaf | Node a (Tree a) (Tree a) deriving (Show, Read, Eq)

singleton :: a -> Tree a
singleton x = Node x Leaf Leaf

treeInsert :: (Ord a) => a -> Tree a -> Tree a
treeInsert x Leaf = singleton x
treeInsert x (Node a left right)
    | x == a = Node x left right
    | x < a  = Node a (treeInsert x left) right
    | x > a  = Node a left (treeInsert x right)

treeElem :: (Ord a) => a -> Tree a -> Bool
treeElem x Leaf = False
treeElem x (Node a left right)
    | x == a = True
    | x < a = treeElem x left
    | x > a = treeElem x right

data TrafficLight = Red | Yellow | Green

instance Eq TrafficLight where
    Red == Red = True
    Green == Green = True
    Yellow == Yellow = True
    _ == _ = False

instance Show TrafficLight where
    show Red = "Red light"
    show Yellow = "Yellow light"
    show Green = "Green light"

class YesNo a where
    yesno :: a -> Bool

instance YesNo Int where
    yesno 0 = False
    yesno _ = True

instance YesNo [a] where
    yesno [] = False
    yesno _ = True

instance YesNo Bool where
    yesno = id

instance YesNo (Maybe a) where
    yesno (Just _) = True
    yesno Nothing = False

instance YesNo (Tree a) where
    yesno Leaf = False
    yesno _ = True

instance YesNo TrafficLight where
    yesno Red = False
    yesno _ = True

yesnoIf :: (YesNo y) => y -> a -> a -> a
yesnoIf yesnoVal yesResult noResult = if yesno yesnoVal then yesResult else noResult

instance Functor Tree where
    fmap f Leaf = Leaf
    fmap f (Node x leftsub rightsub) = Node (f x) (fmap f leftsub) (fmap f rightsub)



--------------------------------------------------------------------------------
-- Input and Output
--------------------------------------------------------------------------------

withFile' :: FilePath -> IOMode -> (Handle -> IO a) -> IO a
withFile' path mode f = do
    handle <- openFile path mode
    result <- f handle
    hClose handle
    return result

randoms' :: (RandomGen g, Random a) => g -> [a]
randoms' gen = let (value, newGen) = random gen in value:(randoms' newGen)

-- finiteRandoms :: (RandomGen g, Random a, Num n, Eq n) => n -> g -> ([a], g)
-- finiteRandoms 0 gen = ([], gen)
-- finiteRandoms n gen =
--     let (value, newGen) = random gen
--         (restOfList, finalGen) = finiteRandoms (n - 1) newGen
--     in  (value:restOfList, finalGen)

finiteRandoms :: (RandomGen g, Random a, Num n, Eq n) => n -> g -> ([a], g)
finiteRandoms 0 gen = ([], gen)
finiteRandoms n gen = (value:restOfList, finalGen)
    where (value, newGen) = random gen
          (restOfList, finalGen) = finiteRandoms (n - 1) newGen



--------------------------------------------------------------------------------
-- Functors and Monoids
--------------------------------------------------------------------------------

data CMaybe a = CNothing | CJust Int a deriving (Show)

-- Not realy a functor.
instance Functor CMaybe where
    fmap f CNothing = CNothing
    fmap f (CJust counter x) = CJust (counter + 1) (f x)

-- sequenceA :: (Applicative f) => [f a] -> f [a]
-- sequenceA [] = pure []
-- sequenceA (x:xs) = (:) <$> x <*> sequenceA xs

sequenceA :: (Applicative f) => [f a] -> f [a]
sequenceA = foldr (liftA2 (:)) (pure [])

newtype Pair b a = Pair { getPair :: (a,b) }

instance Functor (Pair c) where
    fmap f (Pair (x,y)) = Pair (f x, y)

-- lengthCompare :: String -> String -> Ordering
-- lengthCompare x y = if a == EQ then b else a
--     where   a = (length x) `compare` (length y)
--             b = x `compare` y

-- lengthCompare :: String -> String -> Ordering
-- lengthCompare x y = (length x `compare` length y) `mappend` (x `compare` y)

lengthCompare :: String -> String -> Ordering
lengthCompare x y = (length x `compare` length y) `mappend` 
                    (vowels x `compare` vowels y) `mappend` 
                    (x `compare` y)
    where vowels = length . filter (`elem` "aeiou")

instance F.Foldable Tree where
    foldMap f Leaf = mempty
    foldMap f (Node x l r) = F.foldMap f l `mappend`
                             f x           `mappend`
                             F.foldMap f r


testTree = Node 5  
            (Node 3  
                (Node 1 Leaf Leaf)  
                (Node 6 Leaf Leaf)  
            )  
            (Node 9  
                (Node 8 Leaf Leaf)  
                (Node 10 Leaf Leaf)  
            )



--------------------------------------------------------------------------------
-- Monads
--------------------------------------------------------------------------------

type KnightPos = (Int,Int)

moveKnight :: KnightPos -> [KnightPos]
moveKnight (c,r) = do
    (c',r') <- [(c+2,r-1),(c+2,r+1),(c-2,r-1),(c-2,r+1)
               ,(c+1,r-2),(c+1,r+2),(c-1,r-2),(c-1,r+2)
               ]
    guard (c' `elem` [1..8] && r' `elem` [1..8])
    return (c',r')

in3 :: KnightPos -> [KnightPos]
in3 start = return start >>= moveKnight >>= moveKnight >>= moveKnight

canReachIn3 :: KnightPos -> KnightPos -> Bool
canReachIn3 start end = end `elem` in3 start

isBigGang :: Int -> Bool
isBigGang x = x > 9

logNumber :: (Num a, Show a) => a -> W.Writer [String] a
logNumber x = W.writer (x, ["Got number: " ++ show x])

multWithLog :: W.Writer [String] Int
multWithLog = do
    a <- logNumber 3
    b <- logNumber 5
    W.tell ["Gonna multiply these two"]
    return (a*b)

gcd' :: Int -> Int -> Int
gcd' a b
    | b == 0    = a
    | otherwise = gcd' b (a `mod` b)

-- Good loggin order (fast).
gcd'' :: Int -> Int -> W.Writer [String] Int
gcd'' a b
    | b == 0 = do
        W.tell ["Finished with " ++ show a]
        return a
    | otherwise = do
        W.tell [show a ++ " mod " ++ show b ++ " = " ++ show (a `mod` b)]
        gcd'' b (a `mod` b)

-- Bad logging order (slow).
gcdReverse :: Int -> Int -> W.Writer [String] Int
gcdReverse a b
    | b == 0 = do
        W.tell ["Finished with " ++ show a]
        return a
    | otherwise = do
        result <- gcdReverse b (a `mod` b)
        W.tell [show a ++ " mod " ++ show b ++ " = " ++ show (a `mod` b)]
        return result

newtype DiffList a = DiffList { getDiffList :: [a] -> [a] }

toDiffList :: [a] -> DiffList a
toDiffList xs = DiffList (xs++)

fromDiffList :: DiffList a -> [a]
fromDiffList (DiffList f) = f []

instance Monoid (DiffList a) where
    mempty = DiffList (\xs -> [] ++ xs)
    (DiffList f) `mappend` (DiffList g) = DiffList (\xs -> f (g xs))

gcdReverseFast :: Int -> Int -> W.Writer (DiffList String) Int
gcdReverseFast a b
    | b == 0 = do
        W.tell (toDiffList ["Finished with " ++ show a])
        return a
    | otherwise = do
        result <- gcdReverseFast b (a `mod` b)
        W.tell (toDiffList [show a ++ " mod " ++ show b ++ " = " ++ show (a `mod` b)])
        return result

finalCountDownFast :: Int -> W.Writer (DiffList String) ()
finalCountDownFast 0 = do
    W.tell (toDiffList ["0"])
finalCountDownFast x = do
    finalCountDownFast (x - 1)
    W.tell (toDiffList [show x])

finalCountDownSlow :: Int -> W.Writer [String] ()
finalCountDownSlow 0 = do
    W.tell ["0"]
finalCountDownSlow x = do
    finalCountDownSlow (x - 1)
    W.tell [show x]

type Stack = [Int]

-- pop :: Stack -> (Int, Stack)
-- pop (x:xs) = (x,xs)
-- 
-- push :: Int -> Stack -> ((),Stack)
-- push a xs = ((),a:xs)
-- 
-- stackManip :: Stack -> (Int, Stack)
-- stackManip stack = result
--     where
--     ((),newStack1) = push 3 stack
--     (a ,newStack2) = pop newStack1
--     result         = pop newStack2

addStuff :: Int -> Int  
addStuff = do  
    a <- (*2)  
    b <- (+10)  
    return (a+b) 

pop :: State Stack Int
pop = state $ \(x:xs) -> (x,xs)

push :: Int -> State Stack ()
push a = state $ \xs -> ((),a:xs)

stackManip :: State Stack Int
stackManip = do
    push 3
    a <- pop
    pop

stackStuff :: State Stack ()
stackStuff = do
    a <- pop
    if a == 5
        then push 5
        else do
            push 3
            push 8

moreStack :: State Stack ()
moreStack = do
    a <- stackManip
    if a == 100
        then stackStuff
        else return ()


stackyStack :: State Stack ()
stackyStack = do
    stackNow <- get
    if stackNow == [1,2,3]
        then put [8,3,1]
        else put [9,2,1]

randomSt :: (RandomGen g, Random a) => State g a
randomSt = state random

threeCoins :: State StdGen (Bool,Bool,Bool)
threeCoins = do
    a <- randomSt
    b <- randomSt
    c <- randomSt
    return (a,b,c)

keepSmall :: Int -> W.Writer [String] Bool
keepSmall x
    | x < 4 = do
        W.tell ["Keeping " ++ show x]
        return True
    | otherwise = do
        W.tell [show x ++ " is too large, throwing it away"]
        return False

powerset :: [a] -> [[a]]
powerset xs = filterM (\x -> [True, False]) xs

binSmalls :: Int -> Int -> Maybe Int
binSmalls acc x
    | x > 9     = Nothing
    | otherwise = Just (acc + x)

solveRPN :: String -> Double
solveRPN = head . foldl foldingFunction [] . words

foldingFunction :: [Double] -> String -> [Double]
foldingFunction (y:x:xs) "*" = (x * y):xs
foldingFunction (y:x:xs) "+" = (x + y):xs
foldingFunction (y:x:xs) "-" = (x - y):xs
foldingFunction xs numberString = read numberString:xs

solveRPN' :: String -> Maybe Double
solveRPN' st = do
    [result] <- foldM foldingFunction' [] (words st)
    return result

foldingFunction' :: [Double] -> String -> Maybe [Double]
foldingFunction' (y:x:xs) "*" = return((x * y):xs)
foldingFunction' (y:x:xs) "+" = return((x + y):xs)
foldingFunction' (y:x:xs) "-" = return((x - y):xs)
foldingFunction' xs numberString = liftM (:xs) (readMaybe numberString)

readMaybe :: (Read a) => String -> Maybe a
readMaybe st = case reads st of 
    [(x,"")] -> Just x
    _ -> Nothing

inMany :: Int -> KnightPos -> [KnightPos]
inMany x start = return start >>= foldr (<=<) return (replicate x moveKnight)

canReachIn :: Int -> KnightPos -> KnightPos -> Bool
canReachIn x start end = end `elem` inMany x start

newtype Prob a = Prob { getProb :: [(a,Rational)] } deriving Show

instance Functor Prob where
    fmap f (Prob xs) = Prob $ map (\(x,p) -> (f x,p)) xs

thisSituation :: Prob (Prob Char)
thisSituation = Prob
    [( Prob [('a',1%2),('b',1%2)] , 1%4 )
    ,( Prob [('c',1%2),('d',1%2)] , 3%4 )
    ]

flatten :: Prob (Prob a) -> Prob a
flatten (Prob xs) = Prob $ concat $ map multAll xs
    where multAll (Prob innerxs,p) = map (\(x,r) -> (x,p*r)) innerxs

instance Monad Prob where
    return x = Prob [(x,1%1)]
    m >>= f = flatten (fmap f m)
    fail _ = Prob []

data Coin = Heads | Tails deriving (Show, Eq)

coin :: Prob Coin
coin = Prob [(Heads,1%2),(Tails,1%2)]

loadedCoin :: Prob Coin
loadedCoin = Prob [(Heads,1%10),(Tails,9%10)]

flipThree :: Prob Bool
flipThree = do
    a <- coin
    b <- coin
    c <- loadedCoin
    return (all (==Tails) [a,b,c])


