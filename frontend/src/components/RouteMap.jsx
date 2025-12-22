/**
 * Route Map Component
 * Leaflet-based map visualization for CVRP routes
 */

import { useEffect, useMemo } from 'react';
import { MapContainer, TileLayer, Marker, Popup, Polyline, useMap } from 'react-leaflet';
import L from 'leaflet';
import 'leaflet/dist/leaflet.css';

// Fix Leaflet default marker icon issue
delete L.Icon.Default.prototype._getIconUrl;
L.Icon.Default.mergeOptions({
  iconRetinaUrl: 'https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.7.1/images/marker-icon-2x.png',
  iconUrl: 'https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.7.1/images/marker-icon.png',
  shadowUrl: 'https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.7.1/images/marker-shadow.png',
});

// Custom depot icon (red square)
const depotIcon = new L.DivIcon({
  className: 'custom-depot-icon',
  html: `<div style="
    width: 24px;
    height: 24px;
    background-color: #1A1A2E;
    border: 3px solid #DA1E28;
    border-radius: 4px;
    display: flex;
    align-items: center;
    justify-content: center;
    color: white;
    font-size: 12px;
    font-weight: bold;
  ">D</div>`,
  iconSize: [24, 24],
  iconAnchor: [12, 12],
});

// Custom customer icon (teal circle)
const createCustomerIcon = (label, isHighlighted = false) => {
  const bgColor = isHighlighted ? '#FF832B' : '#009D9A';
  return new L.DivIcon({
    className: 'custom-customer-icon',
    html: `<div style="
      width: 28px;
      height: 28px;
      background-color: ${bgColor};
      border: 2px solid white;
      border-radius: 50%;
      display: flex;
      align-items: center;
      justify-content: center;
      color: white;
      font-size: 11px;
      font-weight: bold;
      box-shadow: 0 2px 4px rgba(0,0,0,0.3);
    ">${label}</div>`,
    iconSize: [28, 28],
    iconAnchor: [14, 14],
  });
};

// Component to fit map bounds
function FitBounds({ bounds }) {
  const map = useMap();
  useEffect(() => {
    if (bounds && bounds.length > 0) {
      map.fitBounds(bounds, { padding: [50, 50] });
    }
  }, [map, bounds]);
  return null;
}

// Convert problem coordinates to map coordinates
// We use a simple scaling: x,y -> lat,lng centered around 0,0
function coordToLatLng(x, y, scale = 0.01) {
  return [y * scale, x * scale];
}

export default function RouteMap({
  depot = null,
  customers = [],
  route = [],
  highlightedCustomer = null,
}) {
  // Calculate all positions for bounds
  const allPositions = useMemo(() => {
    const positions = [];
    if (depot) {
      positions.push(coordToLatLng(depot.x, depot.y));
    }
    customers.forEach(c => {
      positions.push(coordToLatLng(c.x, c.y));
    });
    return positions;
  }, [depot, customers]);

  // Create route polyline coordinates
  const routePath = useMemo(() => {
    if (!route || route.length === 0 || !depot) return [];

    return route.map(nodeId => {
      if (nodeId === 0) {
        return coordToLatLng(depot.x, depot.y);
      }
      const customer = customers.find(c => c.id === nodeId);
      if (customer) {
        return coordToLatLng(customer.x, customer.y);
      }
      return null;
    }).filter(Boolean);
  }, [route, depot, customers]);

  // Get visit order for a customer
  const getVisitOrder = (customerId) => {
    const routeWithoutDepot = route.filter(id => id !== 0);
    const index = routeWithoutDepot.indexOf(customerId);
    return index >= 0 ? index + 1 : null;
  };

  return (
    <div className="h-full w-full rounded-lg overflow-hidden border border-[#D1D1E0]">
      <MapContainer
        center={[0, 0]}
        zoom={13}
        className="h-full w-full"
        scrollWheelZoom={true}
      >
        <TileLayer
          attribution='&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a>'
          url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png"
        />

        {/* Fit to bounds */}
        {allPositions.length > 0 && <FitBounds bounds={allPositions} />}

        {/* Route path */}
        {routePath.length > 1 && (
          <Polyline
            positions={routePath}
            color="#0043CE"
            weight={3}
            opacity={0.8}
          />
        )}

        {/* Depot marker */}
        {depot && (
          <Marker
            position={coordToLatLng(depot.x, depot.y)}
            icon={depotIcon}
          >
            <Popup>
              <div className="text-sm">
                <strong>DEPOT</strong>
                <br />
                Location: ({depot.x}, {depot.y})
              </div>
            </Popup>
          </Marker>
        )}

        {/* Customer markers */}
        {customers.map(customer => {
          const visitOrder = getVisitOrder(customer.id);
          const label = visitOrder ? `#${visitOrder}` : customer.id;
          const isHighlighted = highlightedCustomer === customer.id;

          return (
            <Marker
              key={customer.id}
              position={coordToLatLng(customer.x, customer.y)}
              icon={createCustomerIcon(label, isHighlighted)}
            >
              <Popup>
                <div className="text-sm">
                  <strong>Customer {customer.id}</strong>
                  {customer.name && <span> - {customer.name}</span>}
                  <br />
                  Location: ({customer.x}, {customer.y})
                  <br />
                  Demand: {customer.demand} units
                  {visitOrder && (
                    <>
                      <br />
                      <span className="text-[#0043CE] font-medium">
                        Visit Order: #{visitOrder}
                      </span>
                    </>
                  )}
                </div>
              </Popup>
            </Marker>
          );
        })}
      </MapContainer>
    </div>
  );
}
