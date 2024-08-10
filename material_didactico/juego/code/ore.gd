extends Area2D

@export var speed = 100
@export var movment = 100
@export var direction = Vector2i(0, 0)
@export var target_position = Vector2i(0, 0)
@export var current_belt = Vector2i(0,0)
@export var icon = Vector2i(0,0)


func _process(delta):
	position += direction*100*delta
