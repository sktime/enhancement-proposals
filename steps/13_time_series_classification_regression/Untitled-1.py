from abc import ABC, abstractmethod

class AbstrCl(ABC):

    @abstractmethod
    def say_hi(self):
        pass

    def say_bye(self):
        pass


class ByeCl(AbstrCl, ABC):

    def say_bye(self):
        print("bye")

class Test:

    def say_hi(self):
        print("hi")


class Impl(Test, ByeCl):
    def mmm(self):
        print("hi")



###### Code #####
greeter = Impl()
greeter.say_bye()