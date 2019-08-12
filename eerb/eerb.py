import pygame
from random import random
from shapely.geometry import Point, Polygon
from shapely.ops import cascaded_union
import shapely.affinity
from math import sin, cos, pi
from FireflyAlgorithm import FireflyAlgorithm
RED = (255, 0, 0)
GREEN = (0, 255, 0)
BLUE = (0, 0, 255)
 
def degToRad(angle):
    return(pi*angle/180)
 
def drawHexagon(x, y, r):
    # vertices = [[x - r, y - r], [x - r, y + r], [x + r, y + r], [x + r, y - r]] square vertices
    vertices = [(x + cos(degToRad(i*60))*r, y + sin(degToRad(i*60))*r) for i in range(6)]
    pygame.draw.polygon(screen, BLUE, vertices)
 
def drawCircle(x, y, r):
    x, y = int(x), int(y)
    pygame.draw.circle(screen, GREEN, (x, y), r)
 
def drawCity():
    for hexagon in city:
        drawHexagon(hexagon[0], hexagon[1], hex_radious)
 
class BaseStationFitness:
 
    def __init__(self, stations, city_polygon):
        self.stations = stations
        self.minimum_values = [0 for i in range(stations)]
        self.maximum_values = [625 for i in range(stations)]
        self.city_polygon = city_polygon
 
    def minimum_decision_variable_values(self):
        return(self.minimum_values)
    def maximum_decision_variable_values(self):
        return(self.maximum_values)
 
    def objective_function_value(self, bases):
        ans = []
        for i in range(0, len(bases), 2):
            x, y = bases[i], bases[i + 1]
            ans += [Point(x, y).buffer(circle_radious)] # makes a 16-gon polygon to approximate a circle
        covered_area = cascaded_union(ans)
        return(self.city_polygon.intersection(covered_area).area / self.city_polygon.area)
 
# Reading input
f = open("in", "r")
stations, hex_radious, circle_radious, city_hexagons = list(map(int, f.readline().split()))
city, city_polygon = [], []
for i in range(city_hexagons):
    x, y = list(map(int, f.readline().split()))
    city += [[x, y]]
    city_polygon.append(Polygon([(x + cos(degToRad(j*60))*hex_radious, y + sin(degToRad(j*60))*hex_radious) for j in range(6)]))
city_polygon = cascaded_union(city_polygon)
f.close()
 
firefly_algorithm = FireflyAlgorithm(BaseStationFitness(2*stations, city_polygon), 2*stations)
ans = firefly_algorithm.search(100, 700)
bases = ans["best_decision_variable_values"]
print(ans)
 
# Drawing stuff
pygame.init()
width, height = 800, 600
screen = pygame.display.set_mode((width, height))
while True:
    for event in pygame.event.get():
        pass
    input()
   
    screen.fill((255, 255, 255))
    drawCity()
 
    ans = []
    for i in range(0, len(bases), 2):
        x, y = bases[i], bases[i + 1]
        ans += [Point(x, y).buffer(circle_radious)] # makes a 16-gon polygon to approximate a circle
        drawCircle(x, y, circle_radious)
    covered_area = cascaded_union(ans)
    coverage = city_polygon.intersection(covered_area).area / city_polygon.area
    print(coverage)
   
    pygame.display.update()
 
pygame.quit()
