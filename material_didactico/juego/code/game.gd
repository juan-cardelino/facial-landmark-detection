extends Node2D

@export var ore_node: PackedScene
@onready var tilemap = $TileMap

const  ore_leyer = 0
const minor_layer = 1
const module_layer = 2
const belt_layer = 3
const mouse_layer = 4
var minors:Array = []
var belts:Array = []
var full_inventory:Array = []
var units:Array = []
var modo:String = 'minor'
var rotacion:int = 0
var moving:Array = []


# Called when the node enters the scene tree for the first time.
func _ready():
	pass


# Called every frame. 'delta' is the elapsed time since the previous frame.
func _process(_delta):
	for i in units:
		var world_position = i.position
		var map_position = tilemap.local_to_map(world_position)
		if  map_position in belts:
			var tile_atlas_coor = tilemap.get_cell_atlas_coords(belt_layer, map_position)
			var direction = Vector2(1,0).rotated(tile_atlas_coor[0]*PI/4)
			if i not in moving:
				var output_tile_on_map = tilemap.local_to_map(world_position)+Vector2i(direction)
				if tilemap.get_cell_atlas_coords(belt_layer, output_tile_on_map) != Vector2i(-1, -1):
					if output_tile_on_map not in full_inventory:
						full_inventory += [output_tile_on_map]
						i.direction = direction
						i.target_position = world_position+direction*32
						moving += [i]
						i.current_blet = map_position
				elif tilemap.get_cell_atlas_coords(module_layer, output_tile_on_map) != Vector2i(-1, -1):
					i.direction = direction
					full_inventory.remove_at(full_inventory.rfind(i.current_blet))
					moving += [i]
			else:
				if i.position.distance_to(i.target_position) < 1:
					i.position = i.target_position
					i.direction = Vector2(0,0)
					moving.remove_at(moving.rfind(i))
				if map_position != i.current_blet:
					full_inventory.remove_at(full_inventory.rfind(i.current_blet))
					i.current_blet = map_position
		else:
			if i in moving:
				moving.remove_at(moving.rfind(i))
			units.remove_at(units.rfind(i))
			i.queue_free()



func _input(_event):
	if Input.is_action_just_pressed("Seleccionar"):
		if modo == 'minor':
			modo = 'belt'
		elif modo == 'belt':
			modo = 'minor'
		print(modo)
	if Input.is_action_just_pressed("Rotar"):
		rotacion = (rotacion+1)%4
	if Input.is_action_just_pressed("click"):
		var mouse_map_position = tilemap.local_to_map(get_global_mouse_position())
		var ore_tile_atlas_coor = tilemap.get_cell_atlas_coords(ore_leyer, mouse_map_position)
		if modo == 'minor':
			if ore_tile_atlas_coor != Vector2i(-1, -1):
				if tilemap.get_cell_atlas_coords(belt_layer, mouse_map_position) == Vector2i(-1, -1):
					tilemap.set_cell(minor_layer, mouse_map_position, 0, Vector2i(rotacion+ore_tile_atlas_coor[0]*4,1))
					if mouse_map_position not in minors:
						minors += [mouse_map_position]
		elif modo == 'belt':
			if tilemap.get_cell_atlas_coords(minor_layer, mouse_map_position) == Vector2i(-1, -1):
				tilemap.set_cell(belt_layer, mouse_map_position, 0, Vector2i(rotacion*2, 3))
				if mouse_map_position not in belts:
					belts += [mouse_map_position]
	if Input.is_action_just_pressed('click negativo'):
		var mouse_map_position = tilemap.local_to_map(get_global_mouse_position())
		if modo == 'minor':
			if tilemap.get_cell_atlas_coords(minor_layer, mouse_map_position) != Vector2i(-1, -1):
				tilemap.erase_cell(minor_layer, mouse_map_position)
				minors.remove_at(minors.rfind(mouse_map_position))
		elif modo == 'belt':
			if tilemap.get_cell_atlas_coords(belt_layer, mouse_map_position) != Vector2i(-1, -1):
				tilemap.erase_cell(belt_layer, mouse_map_position)
				belts.remove_at(belts.rfind(mouse_map_position))
				if mouse_map_position in full_inventory:
					full_inventory.remove_at(full_inventory.rfind(mouse_map_position))

func _on_ore_timer_timeout():
	for i in minors:
		var tile_atlas_coor = tilemap.get_cell_atlas_coords(minor_layer, i)
		var output_belt_map_coord = i+Vector2i(Vector2(1,0).rotated(tile_atlas_coor[0]*PI/2))
		if tilemap.get_cell_atlas_coords(belt_layer, output_belt_map_coord) != Vector2i(-1, -1):
			if output_belt_map_coord not in full_inventory:
				full_inventory += [output_belt_map_coord]
				var unit = ore_node.instantiate()
				unit.position = output_belt_map_coord*32 + Vector2i(16, 16)
				add_child(unit)
				units += [unit]
