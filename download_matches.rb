require 'json'
require 'set'

MY_PLAYER_ID = 1141532

def get_matches_for(player_id)
  puts "Getting matches for player: #{player_id}."
  `curl http://pwnageddon.local:3000/profiles/#{player_id}/matches`
end

def pathname_for(player_id)
  "./matches/json/#{player_id}.json"
end

def valid_json?(json)
  !JSON.parse(json).empty?
rescue JSON::ParserError
  false
end

def save_json_for(player_id)
  matches = get_matches_for player_id
  if valid_json?(matches)
    File.open(pathname_for(player_id), 'w') { |f| f.write matches }
  else
    puts "Invalid JSON for player: #{player_id}."
  end
end

def valid_file_exists_for?(player_id)
  path = pathname_for player_id
  File.file?(path) && valid_json?(File.read(path))
end

my_match_file = pathname_for(MY_PLAYER_ID)
my_match_data = File.file?(my_match_file) ? File.read(my_match_file) : get_matches_for(MY_PLAYER_ID)
my_matches = JSON.parse my_match_data

my_matches.each_with_index do |match, i|
  puts "\nMatch #{i + 1} / #{my_matches.length}"
  player_ids = match['players'].map { |player| player['playerId'] }
  puts "Players: #{player_ids}"
  player_ids.each do |player_id|
    next if player_id.nil?
    save_json_for(player_id) unless valid_file_exists_for?(player_id)
  end
end
